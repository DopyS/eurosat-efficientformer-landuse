from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eurosat_landuse.config import describe_config, load_config
from src.eurosat_landuse.paths import project_path
from src.eurosat_landuse.predict import predict_pil_image


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install dependencies with `pip install -r requirements.txt`.")
        return

    from PIL import Image

    config = load_config()
    st.set_page_config(page_title="EuroSAT Demo", layout="centered")
    st.title("EuroSAT Land Use Classification")
    st.caption("EfficientFormerV2-S0 EuroSAT image classifier")

    with st.sidebar:
        st.subheader("Run Settings")
        checkpoint_options = available_checkpoints()
        default_checkpoint = "outputs/checkpoints/baseline_100b_best.pt"
        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoint_options,
            index=checkpoint_options.index(default_checkpoint) if default_checkpoint in checkpoint_options else 0,
        )
        top_k = st.slider("Top-k", min_value=1, max_value=5, value=3)
        device = st.selectbox("Device", ["auto", "mps", "cpu"], index=0)
        with st.expander("Config Summary"):
            st.code(describe_config(config))

    uploaded = st.file_uploader("Upload one satellite image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if checkpoint:
            with st.spinner("Running prediction..."):
                predictions = predict_pil_image(
                    config,
                    checkpoint_path=checkpoint,
                    image=image,
                    device_name=device,
                    top_k=top_k,
                )
            st.subheader("Prediction")
            best = predictions[0]
            st.metric(best["class_name"], f"{best['confidence']:.2%}")
            st.table(
                [
                    {
                        "Rank": item["rank"],
                        "Class": item["class_name"],
                        "Confidence": f"{item['confidence']:.2%}",
                    }
                    for item in predictions
                ]
            )
        else:
            st.warning("No checkpoint found under outputs/checkpoints.")
    else:
        st.info("Upload a EuroSAT-style RGB image to run prediction.")


def available_checkpoints() -> list[str]:
    checkpoint_dir = project_path("outputs", "checkpoints")
    if not checkpoint_dir.exists():
        return []
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pt"))
    return [str(path.relative_to(project_path())) for path in checkpoints]


if __name__ == "__main__":
    main()
