from __future__ import annotations


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install dependencies with `pip install -r requirements.txt`.")
        return

    from src.eurosat_landuse.config import describe_config, load_config

    config = load_config()
    st.set_page_config(page_title="EuroSAT Demo", layout="centered")
    st.title("EuroSAT Land Use Classification")
    st.caption("Stage 2 scaffold: interface shell only.")
    st.write(describe_config(config))

    uploaded = st.file_uploader("Upload one satellite image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        st.info("Prediction wiring will be added in the next stage.")
    else:
        st.info("Upload an image to preview the interface layout.")


if __name__ == "__main__":
    main()

