from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas.api.types as ptypes
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# Report Structures

@dataclass
class ValidationIssue:
    rule: str
    column: Optional[str]
    severity: str
    message: str
    count: int


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def summary(self) -> Dict[str, int]:
        result = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        for issue in self.issues:
            result[issue.severity] += 1
        return result


# Base Validator

class BaseValidator(ABC):
    def __init__(self, severity: str = "ERROR"):
        self.severity = severity

    @abstractmethod
    def validate(self, df: pd.DataFrame, report: ValidationReport) -> None:
        pass


# Missing Value Validator

class MissingValueValidator(BaseValidator):
    def __init__(self, threshold: float = 0.1, severity: str = "WARNING"):
        super().__init__(severity)
        self.threshold = threshold

    def validate(self, df: pd.DataFrame, report: ValidationReport) -> None:
        total = len(df)

        for col in df.columns:
            missing = df[col].isna().sum()
            ratio = missing / total

            if ratio > self.threshold:
                report.add_issue(
                    ValidationIssue(
                        rule="MissingValue",
                        column=col,
                        severity=self.severity,
                        message=f"{ratio:.2%} missing values",
                        count=missing,
                    )
                )


# Duplicate Validator

class DuplicateValidator(BaseValidator):
    def __init__(self, subset: Optional[List[str]] = None):
        super().__init__("ERROR")
        self.subset = subset

    def validate(self, df: pd.DataFrame, report: ValidationReport) -> None:
        dup_count = df.duplicated(subset=self.subset).sum()

        if dup_count > 0:
            report.add_issue(
                ValidationIssue(
                    rule="Duplicates",
                    column=None,
                    severity=self.severity,
                    message="Duplicate rows detected",
                    count=int(dup_count),
                )
            )


# Schema Validator

class SchemaValidator(BaseValidator):
    def __init__(self, expected_schema: Dict[str, Any]):
        super().__init__("ERROR")
        self.expected_schema = expected_schema

    def validate(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for col, expected_type in self.expected_schema.items():

            if col not in df.columns:
                report.add_issue(
                    ValidationIssue(
                        rule="Schema",
                        column=col,
                        severity=self.severity,
                        message="Column missing",
                        count=1,
                    )
                )
                continue

            actual_dtype = df[col].dtype

            type_ok = False

            # ---------- SAFE TYPE CHECKS ----------
            if expected_type == "string":
                type_ok = ptypes.is_string_dtype(actual_dtype)

            elif expected_type == "int":
                type_ok = ptypes.is_integer_dtype(actual_dtype)

            elif expected_type == "float":
                type_ok = ptypes.is_float_dtype(actual_dtype)

            elif expected_type == "datetime":
                type_ok = ptypes.is_datetime64_any_dtype(actual_dtype)

            elif expected_type == "bool":
                type_ok = ptypes.is_bool_dtype(actual_dtype)

            else:
                type_ok = True  # fallback

            if not type_ok:
                report.add_issue(
                    ValidationIssue(
                        rule="Schema",
                        column=col,
                        severity=self.severity,
                        message=f"Expected {expected_type}, found {actual_dtype}",
                        count=1,
                    )
                )


# Outlier Validator

class OutlierValidator(BaseValidator):
    def __init__(
        self,
        method: str = "iqr",
        z_threshold: float = 3.0,
        severity: str = "WARNING",
    ):
        super().__init__(severity)
        self.method = method
        self.z_threshold = z_threshold

    def validate(self, df: pd.DataFrame, report: ValidationReport) -> None:
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            series = df[col].dropna()

            if self.method == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = ((series < lower) | (series > upper)).sum()

            elif self.method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = (z_scores > self.z_threshold).sum()

            else:
                raise ValueError("Invalid method. Use 'iqr' or 'zscore'")

            if outliers > 0:
                report.add_issue(
                    ValidationIssue(
                        rule="Outliers",
                        column=col,
                        severity=self.severity,
                        message=f"{outliers} outliers detected using {self.method}",
                        count=int(outliers),
                    )
                )


# Validation Engine

class DataValidator:
    def __init__(self, validators: List[BaseValidator]):
        self.validators = validators

    def run(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()

        for validator in self.validators:
            validator.validate(df, report)

        return report



if __name__ == "__main__":

    print("Loading dataset...")
    df = pd.read_csv(
        "C:\\Users\\demon\\OneDrive\\Desktop\\Assessment\\test_data\\sample_data.csv",
        parse_dates=["timestamp"]
    )

    # Define expected schema (minimal example)
    expected_schema = {
    "transaction_id": "string",
    "customer_id": "string",
    "timestamp": "datetime",
    "quantity": "int",
    "unit_price": "float",
    "total_amount": "float",
}


    # Initialize validators
    validators = [
        MissingValueValidator(threshold=0.1),
        DuplicateValidator(subset=["transaction_id"]),
        SchemaValidator(expected_schema=expected_schema),
        OutlierValidator(method="iqr"),
    ]

    validator = DataValidator(validators=validators)

    print("\nRunning Validation Checks...")
    report = validator.run(df)

    print("\nValidation Summary:")
    print(report.summary())

    print("\nDetailed Issues:")
    for issue in report.issues:
        print(
            f"[{issue.severity}] {issue.rule} "
            f"(Column: {issue.column}) - {issue.message} "
            f"| Count: {issue.count}"
        )

