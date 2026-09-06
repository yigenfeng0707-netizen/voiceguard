"""Convert SenseVoiceSmall encoder to OpenVINO IR for NPU inference.

Key: encoder.forward(xs_pad, ilens) needs tuple input for tracing.
"""

import os, sys, time, numpy as np, torch

MODEL_DIR = "D:/APPs/OpenVINO/demo/local-meeting-minutes/models/SenseVoiceSmall"
OUTPUT_DIR = "D:/APPs/Intel苏州线下比赛/voiceguard/models/SenseVoiceSmall_ov"

print("=== SenseVoice Encoder OpenVINO Conversion ===")

# Step 1: Load model
print("[1/4] Loading SenseVoiceSmall...")
t0 = time.perf_counter()
from funasr import AutoModel

model = AutoModel(model=MODEL_DIR, device="cpu", disable_update=True)
pt_model = model.model
pt_model.eval()
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
print(f"  Model: {pt_model.__class__.__name__}")
print(f"  Encoder: {pt_model.encoder.__class__.__name__}")

# Step 2: Wrap encoder for single-input tracing
print("[2/4] Wrapping encoder for OpenVINO conversion...")


class EncoderWrapper(torch.nn.Module):
    """Wraps SenseVoice encoder to accept (mel_features, lengths) as inputs."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, xs_pad, ilens):
        return self.encoder(xs_pad, ilens)


encoder_wrapper = EncoderWrapper(pt_model.encoder)
encoder_wrapper.eval()

# Create sample inputs matching encoder's expected format
# xs_pad: (batch, frames, mel_bins) - SenseVoice uses 80 mel bins
# ilens: (batch,) - actual frame count per sample
batch_size = 1
num_frames = 100
mel_bins = 80
xs_pad = torch.randn(batch_size, num_frames, mel_bins)
ilens = torch.tensor([num_frames], dtype=torch.long)

# Test forward pass
with torch.no_grad():
    try:
        out = encoder_wrapper(xs_pad, ilens)
        if isinstance(out, tuple):
            print(
                f"  Forward test: output shapes = {[o.shape if hasattr(o, 'shape') else str(o) for o in out]}"
            )
        else:
            print(f"  Forward test: output shape = {out.shape}")
    except Exception as e:
        print(f"  Forward test failed: {e}")
        # Try different mel_bins values
        for mb in [560, 80, 1280, 2576]:
            xs_try = torch.randn(1, num_frames, mb)
            try:
                out = encoder_wrapper(xs_try, ilens)
                print(
                    f"  Forward with mel_bins={mb}: output shape = {out[0].shape if isinstance(out, tuple) else out.shape}"
                )
                mel_bins = mb
                xs_pad = xs_try
                break
            except Exception:
                continue

# Step 3: Convert to OpenVINO IR
print("[3/4] Converting to OpenVINO IR...")
import openvino
from openvino import convert_model

t1 = time.perf_counter()
try:
    ov_model = convert_model(
        encoder_wrapper,
        example_input=(xs_pad, ilens),
        share_weights=False,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xml_path = os.path.join(OUTPUT_DIR, "encoder.xml")
    openvino.save_model(ov_model, xml_path, compress_to_fp16=True)
    print(f"  Converted in {time.perf_counter() - t1:.1f}s")
    print(f"  Saved: {xml_path}")
except Exception as e:
    print(f"  convert_model failed: {e}")
    # Try with torch.jit.trace first
    print("  Trying torch.jit.trace approach...")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(encoder_wrapper, (xs_pad, ilens), strict=False)
        ov_model = convert_model(traced)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        xml_path = os.path.join(OUTPUT_DIR, "encoder.xml")
        openvino.save_model(ov_model, xml_path, compress_to_fp16=True)
        print(f"  Traced + converted in {time.perf_counter() - t1:.1f}s")
        print(f"  Saved: {xml_path}")
    except Exception as e2:
        print(f"  JIT trace also failed: {e2}")
        import traceback

        traceback.print_exc()

# Step 4: Verify and try NPU
print("[4/4] Checking output and testing NPU...")
if os.path.exists(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
        print(f"  {f}: {sz:.1f} MB")

    # Try loading on NPU
    try:
        core = openvino.Core()
        devices = core.available_devices
        print(f"  Available devices: {devices}")

        # Read and compile for NPU
        model_ir = core.read_model(xml_path)

        # Try NPU first
        for dev in ["NPU", "GPU", "CPU"]:
            try:
                compiled = core.compile_model(model_ir, dev)
                # Test inference
                input_data = (
                    np.random.randn(1, num_frames, mel_bins).astype(np.float32),
                    np.array([num_frames], dtype=np.int64),
                )
                result = compiled(input_data)
                output_keys = list(result.keys())
                output_shapes = {k: result[k].shape for k in output_keys}
                print(f"  {dev} inference OK: outputs = {output_shapes}")
                break
            except Exception as e:
                print(f"  {dev} failed: {e}")
    except Exception as e:
        print(f"  NPU test failed: {e}")
        import traceback

        traceback.print_exc()
else:
    print("  No output directory created")

print("=== Done ===")
