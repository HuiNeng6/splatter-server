import re


def rename_for_domain_upload(filename: str) -> str | None:
    """Map generated global partition outputs to domain upload naming."""
    if filename == "combined_splat.ply":
        return None

    splat_match = re.match(r"combined_splat_partition_(.+)\.splat$", filename)
    if splat_match:
        suffix = splat_match.group(1)
        return f"splat_partition_{suffix}.splat_partition"

    sog_match = re.match(r"combined_splat_partition_(.+)\.sog$", filename)
    if sog_match:
        suffix = sog_match.group(1)
        return f"splat_partition_sog_{suffix}.splat_partition_sog"

    return None
