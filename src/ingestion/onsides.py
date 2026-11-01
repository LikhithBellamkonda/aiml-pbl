from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.common import IngestedSource, drug_name_key, is_combination_name


LOGGER = logging.getLogger("pregnancy_dataset_ingestion.onsides")
RELEASE = "v3.1.1"
SOURCE_URL = f"https://github.com/tatonetti-lab/onsides/releases/download/{RELEASE}/onsides-v3.1.1.zip"
EXPECTED_SHA256 = "7739c73336786c92e73ca383af92d3f2814caf5e94206e08475b86d5a5ec22b9"
EFFECT_COLUMNS = (
    "product_label_id",
    "effect_id",
    "label_section",
    "effect_meddra_id",
    "match_method",
    "pred0",
    "pred1",
)
SALT_FORM_TERMS = (
    "acetate",
    "besylate",
    "bromide",
    "chloride",
    "citrate",
    "fumarate",
    "hydrobromide",
    "hydrochloride",
    "iodide",
    "maleate",
    "mesylate",
    "monohydrate",
    "phosphate",
    "potassium",
    "sodium",
    "succinate",
    "sulfate",
    "tartrate",
)


def _salt_form_differs(product_name: str, ingredient_name: str) -> bool:
    product = drug_name_key(product_name).split()
    ingredient = set(drug_name_key(ingredient_name).split())
    return any(term in product and term not in ingredient for term in SALT_FORM_TERMS)


def acquire(destination: Path) -> Path:
    from src.ingestion.common import download_file

    return download_file(SOURCE_URL, destination, expected_sha256=EXPECTED_SHA256)


def _read_member(archive: zipfile.ZipFile, name: str, **kwargs: Any) -> pd.DataFrame:
    with archive.open(name) as stream:
        return pd.read_csv(stream, **kwargs)


def ingest(path: Path, *, chunksize: int = 250_000) -> IngestedSource:
    path = Path(path)
    if not path.is_file():
        return IngestedSource(
            "onsides_general_adr",
            False,
            status="source_unavailable",
            source_url=SOURCE_URL,
            source_version=RELEASE,
            source_files=[str(path)],
        )

    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().casefold() != EXPECTED_SHA256:
            raise ValueError("Local OnSIDES archive checksum differs from release v3.1.1.")
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {
                "csv/product_label.csv",
                "csv/product_adverse_effect.csv",
                "csv/vocab_meddra_adverse_effect.csv",
                "csv/vocab_rxnorm_product.csv",
                "csv/vocab_rxnorm_ingredient.csv",
                "csv/vocab_rxnorm_ingredient_to_product.csv",
                "csv/product_to_rxnorm.csv",
            }
            missing = required - names
            if missing:
                raise ValueError(f"OnSIDES release is missing required tables: {sorted(missing)}")

            labels = _read_member(
                archive,
                "csv/product_label.csv",
                usecols=["label_id", "source", "source_product_name", "source_product_id", "source_label_url"],
            )
            products = _read_member(
                archive,
                "csv/product_to_rxnorm.csv",
                usecols=["label_id", "rxnorm_product_id"],
                dtype={"rxnorm_product_id": "string"},
            )
            vocabulary = _read_member(
                archive,
                "csv/vocab_rxnorm_product.csv",
                usecols=["rxnorm_id", "rxnorm_name", "rxnorm_term_type"],
                dtype={"rxnorm_id": "string"},
            )
            ingredients = _read_member(
                archive,
                "csv/vocab_rxnorm_ingredient.csv",
                usecols=["rxnorm_id", "rxnorm_name", "rxnorm_term_type"],
                dtype={"rxnorm_id": "string"},
            )
            ingredient_links = _read_member(
                archive,
                "csv/vocab_rxnorm_ingredient_to_product.csv",
                usecols=["product_id", "ingredient_id"],
                dtype={"product_id": "string", "ingredient_id": "string"},
            )
            meddra = _read_member(
                archive,
                "csv/vocab_meddra_adverse_effect.csv",
                usecols=["meddra_id", "meddra_name", "meddra_term_type"],
            )

            label_products = labels.merge(products, on="label_id", how="inner")
            label_products = label_products.merge(
                vocabulary.rename(
                    columns={
                        "rxnorm_id": "rxnorm_product_id",
                        "rxnorm_name": "rxnorm_product_name",
                        "rxnorm_term_type": "rxnorm_term_type",
                    }
                ),
                on="rxnorm_product_id",
                how="inner",
            )
            label_products = label_products.drop_duplicates(
                ["label_id", "rxnorm_product_id"]
            )
            ingredient_details = ingredient_links.merge(
                ingredients.rename(
                    columns={
                        "rxnorm_id": "ingredient_id",
                        "rxnorm_name": "ingredient_name",
                        "rxnorm_term_type": "ingredient_term_type",
                    }
                ),
                on="ingredient_id",
                how="left",
            )
            ingredients_by_product = {
                product_id: sorted(
                    {
                        (str(row.ingredient_id), str(row.ingredient_name))
                        for row in group.itertuples(index=False)
                        if pd.notna(row.ingredient_name)
                    }
                )
                for product_id, group in ingredient_details.groupby(
                    "product_id", sort=False
                )
            }

            with tempfile.TemporaryDirectory(prefix="onsides-ingestion-") as temp_dir:
                database_path = Path(temp_dir) / "aggregate.sqlite3"
                with contextlib.closing(sqlite3.connect(database_path)) as connection:
                    connection.execute(
                        """
                        CREATE TABLE evidence (
                            rxnorm_product_id TEXT NOT NULL,
                            label_section TEXT NOT NULL,
                            meddra_id INTEGER NOT NULL,
                            term_type TEXT NOT NULL,
                            source_record_count INTEGER NOT NULL,
                            product_label_count INTEGER NOT NULL,
                            max_pred0 REAL,
                            max_pred1 REAL,
                            sample_product_label_id TEXT,
                            sample_effect_id TEXT,
                            match_methods TEXT,
                            PRIMARY KEY (rxnorm_product_id, label_section, meddra_id, term_type)
                        )
                        """
                    )

                    with archive.open("csv/product_adverse_effect.csv") as effect_stream:
                        chunk_iterator = pd.read_csv(
                            effect_stream,
                            usecols=list(EFFECT_COLUMNS),
                            chunksize=chunksize,
                            low_memory=False,
                        )
                        for chunk in chunk_iterator:
                            expanded = chunk.merge(
                                label_products,
                                left_on="product_label_id",
                                right_on="label_id",
                                how="inner",
                            )
                            if expanded.empty:
                                continue
                            expanded["effect_meddra_id"] = pd.to_numeric(
                                expanded["effect_meddra_id"], errors="coerce"
                            )
                            expanded = expanded.dropna(subset=["effect_meddra_id"])
                            expanded = expanded.merge(
                                meddra.rename(
                                    columns={
                                        "meddra_id": "effect_meddra_id",
                                        "meddra_term_type": "mapped_meddra_term_type",
                                    }
                                ),
                                on="effect_meddra_id",
                                how="left",
                            )
                            grouped = (
                                expanded.groupby(
                                    [
                                        "rxnorm_product_id",
                                        "label_section",
                                        "effect_meddra_id",
                                        "mapped_meddra_term_type",
                                    ],
                                    dropna=False,
                                )
                                .agg(
                                    source_record_count=("effect_id", "size"),
                                    product_label_count=("product_label_id", "nunique"),
                                    max_pred0=("pred0", "max"),
                                    max_pred1=("pred1", "max"),
                                    sample_product_label_id=("product_label_id", "first"),
                                    sample_effect_id=("effect_id", "first"),
                                    match_methods=(
                                        "match_method",
                                        lambda values: ";".join(
                                            sorted({str(value) for value in values.dropna()})
                                        ),
                                    ),
                                )
                                .reset_index()
                            )
                            connection.executemany(
                                """
                                INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(rxnorm_product_id, label_section, meddra_id, term_type)
                                DO UPDATE SET
                                    source_record_count = source_record_count + excluded.source_record_count,
                                    product_label_count = product_label_count + excluded.product_label_count,
                                    max_pred0 = MAX(max_pred0, excluded.max_pred0),
                                    max_pred1 = MAX(max_pred1, excluded.max_pred1)
                                """,
                                [
                                    (
                                        str(row.rxnorm_product_id),
                                        str(row.label_section),
                                        int(row.effect_meddra_id),
                                        str(row.mapped_meddra_term_type),
                                        int(row.source_record_count),
                                        int(row.product_label_count),
                                        None if pd.isna(row.max_pred0) else float(row.max_pred0),
                                        None if pd.isna(row.max_pred1) else float(row.max_pred1),
                                        str(row.sample_product_label_id),
                                        str(row.sample_effect_id),
                                        str(row.match_methods),
                                    )
                                    for row in grouped.itertuples(index=False)
                                ],
                            )

                    connection.commit()
                    aggregates = pd.read_sql_query(
                        "SELECT * FROM evidence ORDER BY rxnorm_product_id, label_section, meddra_id",
                        connection,
                    )

            product_names = vocabulary.set_index("rxnorm_id")["rxnorm_name"].to_dict()
            term_names = meddra.set_index("meddra_id")["meddra_name"].to_dict()
            term_types = meddra.set_index("meddra_id")["meddra_term_type"].to_dict()
            label_names = labels.set_index("label_id")["source_product_name"].to_dict()
            label_urls = labels.set_index("label_id")["source_label_url"].to_dict()
            label_sources = labels.set_index("label_id")["source"].to_dict()
            label_product_ids = labels.set_index("label_id")["source_product_id"].to_dict()
            sample_label_product_names = label_products.drop_duplicates("label_id").set_index(
                "label_id"
            )["rxnorm_product_name"].to_dict()
            aggregates["drug_name"] = aggregates["rxnorm_product_id"].map(product_names)
            aggregates["drug_name_key"] = aggregates["drug_name"].map(drug_name_key)
            aggregates["source_identifier"] = (
                "RxNorm:" + aggregates["rxnorm_product_id"].astype(str)
            )
            aggregates["drug_id"] = aggregates["source_identifier"]
            aggregates["source_dataset"] = "OnSIDES"
            aggregates["source_release"] = RELEASE
            aggregates["source_record_id"] = (
                aggregates["sample_product_label_id"].astype(str)
                + ":"
                + aggregates["meddra_id"].astype(str)
                + ":"
                + aggregates["label_section"]
            )
            aggregates["source_product_name"] = aggregates[
                "sample_product_label_id"
            ].map(label_names)
            aggregates["source_label_url_sample"] = aggregates[
                "sample_product_label_id"
            ].map(label_urls)
            aggregates["source_label_source_sample"] = aggregates[
                "sample_product_label_id"
            ].map(label_sources)
            aggregates["source_product_id_sample"] = aggregates[
                "sample_product_label_id"
            ].map(label_product_ids)
            aggregates["rxnorm_product_name"] = aggregates[
                "rxnorm_product_id"
            ].map(product_names)
            ingredient_details_by_product = aggregates["rxnorm_product_id"].map(
                ingredients_by_product
            )
            aggregates["generic_ingredient_ids"] = ingredient_details_by_product.map(
                lambda values: json.dumps(
                    [ingredient_id for ingredient_id, _ in values]
                    if isinstance(values, list)
                    else [],
                    ensure_ascii=False,
                )
            )
            aggregates["generic_ingredient_names"] = ingredient_details_by_product.map(
                lambda values: json.dumps(
                    [name for _, name in values]
                    if isinstance(values, list)
                    else [],
                    ensure_ascii=False,
                )
            )
            aggregates["generic_ingredient_name"] = ingredient_details_by_product.map(
                lambda values: values[0][1]
                if isinstance(values, list) and len(values) == 1
                else ""
            )
            aggregates["source_ingredient_count"] = ingredient_details_by_product.map(
                lambda values: len(values) if isinstance(values, list) else 0
            )
            aggregates["salt_form_review_required"] = aggregates.apply(
                lambda row: bool(
                    row["generic_ingredient_name"]
                    and _salt_form_differs(
                        str(row["drug_name"]),
                        str(row["generic_ingredient_name"]),
                    )
                ),
                axis=1,
            )
            aggregates["generic_ingredient_mapping_allowed"] = (
                aggregates["source_ingredient_count"].eq(1)
                & ~aggregates["salt_form_review_required"]
            )
            aggregates["meddra_name"] = aggregates["meddra_id"].map(term_names)
            aggregates["meddra_term_type"] = aggregates["meddra_id"].map(term_types)
            aggregates["sample_label_rxnorm_name"] = aggregates[
                "sample_product_label_id"
            ].map(sample_label_product_names)
            aggregates["outcome_category"] = ""
            aggregates["outcome_term_original"] = aggregates["meddra_name"]
            aggregates["outcome_value_original"] = ""
            aggregates["evidence_status"] = "regulatory_label_adverse_effect_association"
            aggregates["evidence_granularity"] = "drug_product_label_term_association"
            aggregates["trimester"] = "unknown"
            aggregates["gestational_weeks"] = pd.NA
            aggregates["is_combination"] = (
                aggregates["source_ingredient_count"].gt(1)
                | aggregates["drug_name"].map(is_combination_name)
            )
            aggregates["source_url"] = SOURCE_URL
            aggregates["source_file"] = path.name
            aggregates["source_product_name"] = aggregates["source_product_name"].fillna(
                aggregates["rxnorm_product_name"]
            )

            return IngestedSource(
                "onsides_general_adr",
                True,
                aggregates,
                status="available",
                source_url=SOURCE_URL,
                source_version=RELEASE,
                source_files=[str(path)],
            )
    except Exception as error:
        LOGGER.exception("Failed parsing OnSIDES release %s", path)
        return IngestedSource(
            "onsides_general_adr",
            True,
            status="parse_error",
            source_url=SOURCE_URL,
            source_version=RELEASE,
            error=str(error),
            source_files=[str(path)],
        )
