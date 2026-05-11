# Copyright (c) 2024 Microsoft Corporation.
# Modifications Copyright (c) 2026 Sougato
# Licensed under the MIT License

"""All the steps to transform final entities."""

import pandas as pd

from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS
from graphrag.index.utils.hashing import gen_sha512_hash


def finalize_community_reports(
    reports: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final community reports."""
    if reports.empty or "community" not in reports.columns:
        return pd.DataFrame(columns=COMMUNITY_REPORTS_FINAL_COLUMNS)

    # Merge with communities to add shared fields
    community_reports = reports.merge(
        communities.loc[:, ["community", "parent", "children", "size", "period"]],
        on="community",
        how="left",
        copy=False,
    )

    if community_reports.empty:
        return pd.DataFrame(columns=COMMUNITY_REPORTS_FINAL_COLUMNS)

    community_reports["community"] = community_reports["community"].astype(int)
    community_reports["human_readable_id"] = community_reports["community"]
    community_reports["id"] = community_reports.apply(
        lambda row: gen_sha512_hash(row, ["full_content"]), axis=1
    )

    for column in COMMUNITY_REPORTS_FINAL_COLUMNS:
        if column not in community_reports.columns:
            community_reports[column] = None

    return community_reports.loc[
        :,
        COMMUNITY_REPORTS_FINAL_COLUMNS,
    ]
