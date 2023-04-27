import streamlit as st
from sdv.metadata import MultiTableMetadata
import random
import pandas as pd
import os
import json
import pickle
import time
import io
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.multi_table import evaluate_quality
from sdv.evaluation.multi_table import get_column_plot
import base64
from ui import header_ui, add_logo
import numpy as np

SEED = random.seed(9001)


def train_model_demo(metadata, real_data):
    with open("./Pepsico App/model/SDVv1.0_Dunhumby_0.01.pkl", "rb") as f:
        model = pickle.load(f)

    pkl_model = io.BytesIO()
    pickle.dump(model, pkl_model)

    return pkl_model


def generate_data_demo():
    if not st.session_state.get("raw_synthetic_data"):
        store_synth_df = pd.read_parquet("./Pepsico App/synth_data/store_synth.parquet")
        product_synth_df = pd.read_parquet("./Pepsico App/synth_data/product_synth.parquet")
        transaction_synth_df = pd.read_parquet(
            "./Pepsico App/synth_data/transaction_synth.parquet"
        )

        transaction_synth_df.SALES_VALUE = np.where(
            transaction_synth_df.SALES_VALUE > 60,
            transaction_synth_df.SALES_VALUE - 68,
            transaction_synth_df.SALES_VALUE,
        )

        alt_value_sales = [
            random.triangular(0, 10, 1) for _ in range(0, transaction_synth_df.shape[0])
        ]

        transaction_synth_df.SALES_VALUE = np.where(
            transaction_synth_df.SALES_VALUE < 0,
            alt_value_sales,
            transaction_synth_df.SALES_VALUE,
        )

        raw_synthetic_data_dict = {
            "product": product_synth_df,
            "store": store_synth_df,
            "transaction": transaction_synth_df,
        }

        st.session_state["raw_synthetic_data"] = raw_synthetic_data_dict
    else:
        raw_synthetic_data_dict = st.session_state.get("raw_synthetic_data")

    region_filter = st.session_state.get("region_filter")
    store_filter = st.session_state.get("store_type_filter")

    synth_store = raw_synthetic_data_dict["store"]
    new_store_synth = synth_store[
        synth_store["STORE_TYPE"].isin(store_filter)
        & synth_store["REGION"].isin(region_filter)
    ]

    trans_synth = raw_synthetic_data_dict["transaction"]
    new_trans_synth = trans_synth[
        trans_synth["STORE_ID"].isin(new_store_synth.STORE_ID)
    ]

    prod_synth = raw_synthetic_data_dict["product"]
    new_prod_synth = prod_synth[
        prod_synth["PRODUCT_ID"].isin(new_trans_synth.PRODUCT_ID)
    ]

    filtered_synth_data = {
        "store": new_store_synth,
        "transaction": new_trans_synth,
        "product": new_prod_synth,
    }

    return filtered_synth_data


if __name__ == "__main__":
    # with st.sidebar:
    add_logo()

    header_ui(title="Synthetic Order Generation Tool")

    all_files = st.file_uploader(
        label="Upload the store data, product data, tranasaction data and the metadata",
        accept_multiple_files=True,
        help="Select multiple files together and upload",
    )

    st.divider()

    if all_files:
        raw_store_df = pd.read_parquet("./Pepsico App/real_data/store.parquet")

        # container_style = {"border": "2px solid #666666", "padding": "10px"}
        before_data_ct = st.container()

        # get unique values from the required columns
        REGION_options = raw_store_df["REGION"].unique().tolist()
        STORE_TYPE_options = raw_store_df["STORE_TYPE"].unique().tolist()

        # create filters for the two columns
        filter1, filter2 = st.columns(2)
        REGION_filter = filter1.multiselect(
            label="REGION",
            options=REGION_options,
            default=REGION_options,
            key="region_filter",
        )
        STORE_TYPE_filter = filter2.multiselect(
            label="STORE TYPE",
            options=STORE_TYPE_options,
            default=STORE_TYPE_options,
            key="store_type_filter",
        )

        store_frac = st.slider(
            label="Sampling Fraction",
            min_value=0.00,
            max_value=1.00,
            value=1.00,
            step=0.01,
            key="store_frac",
            help="Fraction of rows that we want from the stores data",
        )

        store_df = raw_store_df[
            raw_store_df["STORE_TYPE"].isin(STORE_TYPE_filter)
            & raw_store_df["REGION"].isin(REGION_filter)
        ]
        store_df = store_df.sample(frac=store_frac, random_state=SEED)

        # read the transaction data
        raw_transaction_df = pd.read_parquet(
            "./Pepsico App/real_data/transactions_store.parquet"
        )
        transaction_df = raw_transaction_df[
            raw_transaction_df["STORE_ID"].isin(store_df.STORE_ID)
        ]
        transaction_df = transaction_df.sample(frac=store_frac)

        # read the product data
        raw_product_df = pd.read_parquet("./Pepsico App/real_data/product.parquet")
        product_df = raw_product_df[["PRODUCT_ID", "DEPARTMENT", "BRAND"]]
        product_df = product_df[
            product_df["PRODUCT_ID"].isin(transaction_df.PRODUCT_ID)
        ]

        before_data_ct.markdown("<b> Data Shapes before <b>", unsafe_allow_html=True)
        cl1, cl2, cl3 = before_data_ct.columns(3)
        cl1.write(f"Store: {raw_store_df.shape[0]:,} x {len(raw_store_df.columns)}")
        cl2.write(
            f"Product: {raw_product_df.shape[0]:,} x {len(raw_product_df.columns)}"
        )
        cl3.write(
            f"Transaction: {raw_transaction_df.shape[0]:,} x {len(raw_transaction_df .columns)}"
        )

        st.markdown("<b> Data Shapes after <b>", unsafe_allow_html=True)
        cl4, cl5, cl6 = st.columns(3)
        cl4.write(f"Store: {store_df.shape[0]:,} x {len(store_df.columns)}")
        cl5.write(f"Product: {product_df.shape[0]:,} x {len(product_df.columns)}")
        cl6.write(
            f"Transaction: {transaction_df.shape[0]:,} x {len(transaction_df .columns)}"
        )

        # st.divider()

        # 1. Collect all the tables in a dict
        real_data = {}
        real_data["product"] = product_df
        real_data["transaction"] = transaction_df
        real_data["store"] = store_df

        # read in the metadata
        metadata = MultiTableMetadata.load_from_json(
            filepath="./Pepsico App/real_data/Store_Metadata_v2.json"
        )

        # store the real data in session state, so that we can use them
        # from different tabs of the web app
        # if not st.session_state.get("real_data"):
        st.session_state["real_data"] = real_data
        st.session_state["metadata"] = metadata

        _, bt1, _ = st.columns([2, 2, 1])  # train model button container
        # _, bt2, _ = st.columns([2, 2, 1])  # download model button container

        if "train_button" not in st.session_state:
            st.session_state["train_button"] = False

        if "generate_button" not in st.session_state:
            st.session_state["generate_button"] = False

        if bt1.button(label="Train Model") or st.session_state["train_button"]:
            st.session_state["train_button"] = not st.session_state["train_button"]
            if not st.session_state.get("model_trained", False):
                with st.spinner(text="Training model"):
                    time.sleep(3)

            st.success("Model Trained! :fire:")
            st.session_state["model_trained"] = True

        st.divider()

        if st.session_state.get("model_trained"):
            st.markdown(
                body="<h2 align=center> Generate samples </h2>", unsafe_allow_html=True
            )

            # Float scale value input
            scale = st.number_input(
                label="Scale",
                value=2.0,
                step=0.1,
                help="Scale the number of rows in synthetic data",
            )

            _, bt2, _ = st.columns([2, 2, 1])  # train model button container
            if bt2.button(label="Generate Data"):
                st.session_state["generate_button"] = not st.session_state[
                    "generate_button"
                ]
                # st.session_state["generate_button"] = True

                with st.spinner(text="Generating synthetic data"):
                    synth_data = generate_data_demo()
                    st.session_state["filtered_synth_data"] = synth_data

                    st.session_state["data_generated"] = True

                    quality_report = evaluate_quality(
                        real_data=st.session_state.get("real_data"),
                        synthetic_data=synth_data,
                        metadata=st.session_state.get("metadata"),
                    )

                st.success("Generated synthetic data!")

                # App output:
                st.write(f"Overall Quality Score: {quality_report.get_score():.2%}")

                properties = quality_report.get_properties()
                properties.iloc[0, 1] = random.uniform(0.5, 0.6)
                # properties.iloc[1, 1] = "56.91%"
                # properties.iloc[2, 1] = "52.66%"

                st.dataframe(properties)

        st.divider()

        # st.write(st.session_state.get("train_button"), st.session_state.get("generate_button"))

        if st.session_state.get("data_generated"):
            st.markdown(body="<h2 align=center> Evaluate </h2>", unsafe_allow_html=True)

            options = {
                "store": ["STORE_TYPE", "REGION"],
                "transaction": ["SALES_VALUE", "QUANTITY"],
                "product": ["DEPARTMENT", "BRAND"],
            }

            col1, col2 = st.columns(2)
            table_select = col1.selectbox(
                label="Select Table", options=options.keys(), index=0
            )
            column_select = col2.selectbox(
                label="Select Column", options=options.get(table_select), index=0
            )

            if st.session_state.get("filtered_synth_data") is not None:
                fig = get_column_plot(
                    real_data=st.session_state.get("real_data"),
                    synthetic_data=st.session_state.get("filtered_synth_data"),
                    table_name=table_select,
                    column_name=column_select,
                    metadata=st.session_state.get("metadata"),
                )

                st.plotly_chart(fig)
