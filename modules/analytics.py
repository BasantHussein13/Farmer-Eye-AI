
import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    BASE_DIR /
    "database" /
    "farmer_eye.db"
)


def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def get_prediction_history(
    limit=None
):

    connection = (
        get_connection()
    )

    query = """
    SELECT *
    FROM prediction_history
    ORDER BY id DESC
    """

    params = ()

    if limit is not None:

        query += (
            " LIMIT ?"
        )

        params = (
            int(
                limit
            ),
        )

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=params
    )

    connection.close()

    return dataframe


def get_general_statistics():

    dataframe = (
        get_prediction_history()
    )

    if dataframe.empty:

        return {
            "total_images": 0,
            "healthy_images": 0,
            "diseased_images": 0,
            "uncertain_images": 0,
            "average_confidence": 0.0,
            "average_analysis_time": 0.0,
            "total_detected_objects": 0,
            "feedback_accuracy": None,
        }

    correct = int(
        (
            dataframe[
                "user_feedback"
            ]
            ==
            "Correct"
        ).sum()
    )

    incorrect = int(
        (
            dataframe[
                "user_feedback"
            ]
            ==
            "Incorrect"
        ).sum()
    )

    feedback_total = (
        correct +
        incorrect
    )

    return {
        "total_images":
            len(
                dataframe
            ),

        "healthy_images":
            int(
                (
                    dataframe[
                        "status"
                    ]
                    ==
                    "Healthy"
                ).sum()
            ),

        "diseased_images":
            int(
                (
                    dataframe[
                        "status"
                    ]
                    ==
                    "Diseased"
                ).sum()
            ),

        "uncertain_images":
            int(
                (
                    dataframe[
                        "status"
                    ]
                    ==
                    "Uncertain"
                ).sum()
            ),

        "average_confidence":
            float(
                dataframe[
                    "confidence"
                ]
                .fillna(0)
                .mean()
            ),

        "average_analysis_time":
            float(
                dataframe[
                    "analysis_time"
                ]
                .fillna(0)
                .mean()
            ),

        "total_detected_objects":
            int(
                dataframe[
                    "detected_objects"
                ]
                .fillna(0)
                .sum()
            ),

        "feedback_accuracy":
            (
                correct /
                feedback_total
                if feedback_total
                else None
            ),
    }


def get_disease_distribution():

    connection = (
        get_connection()
    )

    dataframe = pd.read_sql_query(
        """
        SELECT
            disease_name,
            COUNT(*) AS images,
            AVG(confidence)
                AS average_confidence
        FROM prediction_history
        WHERE
            status = 'Diseased'
            AND disease_name IS NOT NULL
        GROUP BY disease_name
        ORDER BY images DESC
        """,
        connection
    )

    connection.close()

    return dataframe


def update_prediction_feedback(
    prediction_id,
    feedback
):

    connection = (
        get_connection()
    )

    connection.execute(
        """
        UPDATE prediction_history
        SET user_feedback = ?
        WHERE id = ?
        """,
        (
            feedback,
            int(
                prediction_id
            )
        )
    )

    connection.commit()
    connection.close()


def update_prediction_notes(
    prediction_id,
    notes
):

    connection = (
        get_connection()
    )

    connection.execute(
        """
        UPDATE prediction_history
        SET notes = ?
        WHERE id = ?
        """,
        (
            notes,
            int(
                prediction_id
            )
        )
    )

    connection.commit()
    connection.close()
