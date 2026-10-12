def build_kernel_to_channels_map(channel_kernel_map):
    """Group a cache's channel_kernel_map by topology: {"kings": [0, ..., 44], ...}, indices sorted."""
    kernel_to_channels = {}
    for entry in channel_kernel_map:
        kernel_name = entry["kernel"]
        if kernel_name not in kernel_to_channels:
            kernel_to_channels[kernel_name] = []
        kernel_to_channels[kernel_name].append(entry["channel"])
    for name in kernel_to_channels:
        kernel_to_channels[name].sort()
    return kernel_to_channels
