import torch
import warp as wp


def warp_stream_from_torch() -> wp.Stream | None:
    if not torch.cuda.is_available():
        return None
    torch_stream: torch.cuda.Stream = torch.cuda.current_stream()
    warp_stream: wp.Stream = wp.stream_from_torch(torch_stream)
    return warp_stream
