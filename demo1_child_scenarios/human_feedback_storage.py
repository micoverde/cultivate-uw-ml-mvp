#!/usr/bin/env python3
"""
Human Feedback Storage System for ML Model Training
Stores and manages human verification/feedback data for OEQ/CEQ classification
"""

import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

class HumanFeedbackStorage:
    def __init__(self, db_path: str = "human_feedback.db", json_backup_path: str = "human_feedback_backup.json"):
        self.db_path = db_path
        self.json_backup_path = json_backup_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with feedback table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS human_feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                scenario_id INTEGER,
                user_response TEXT NOT NULL,
                ml_prediction TEXT NOT NULL,
                ml_confidence REAL NOT NULL,
                human_label TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                feedback_type TEXT NOT NULL,
                session_id TEXT,
                additional_notes TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                total_feedback INTEGER,
                correct_predictions INTEGER,
                incorrect_predictions INTEGER,
                true_positives INTEGER,
                true_negatives INTEGER,
                false_positives INTEGER,
                false_negatives INTEGER,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL
            )
        ''')

        conn.commit()
        conn.close()

    def store_feedback(self,
                      scenario_id: int,
                      user_response: str,
                      ml_prediction: str,
                      ml_confidence: float,
                      human_label: str,
                      session_id: Optional[str] = None,
                      additional_notes: Optional[str] = None) -> str:
        """Store human feedback for a prediction"""

        feedback_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        is_correct = (ml_prediction.upper() == human_label.upper())

        # Determine feedback type (TP, TN, FP, FN)
        feedback_type = self._determine_feedback_type(ml_prediction, human_label)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO human_feedback
            (id, timestamp, scenario_id, user_response, ml_prediction, ml_confidence,
             human_label, is_correct, feedback_type, session_id, additional_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id, timestamp, scenario_id, user_response, ml_prediction,
            ml_confidence, human_label, is_correct, feedback_type, session_id, additional_notes
        ))

        conn.commit()
        conn.close()

        # Also backup to JSON
        self._backup_to_json()

        # Update performance metrics
        self._update_performance_metrics()

        return feedback_id

    def _determine_feedback_type(self, ml_prediction: str, human_label: str) -> str:
        """Determine if feedback is TP, TN, FP, or FN"""
        ml_pred = ml_prediction.upper()
        human_lbl = human_label.upper()

        if ml_pred == "OEQ" and human_lbl == "OEQ":
            return "TP"  # True Positive
        elif ml_pred == "CEQ" and human_lbl == "CEQ":
            return "TN"  # True Negative (CEQ is negative class)
        elif ml_pred == "OEQ" and human_lbl == "CEQ":
            return "FP"  # False Positive
        elif ml_pred == "CEQ" and human_lbl == "OEQ":
            return "FN"  # False Negative
        else:
            return "UNKNOWN"

    def get_feedback_summary(self) -> Dict:
        """Get summary statistics of all feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get counts by feedback type
        cursor.execute('''
            SELECT feedback_type, COUNT(*)
            FROM human_feedback
            GROUP BY feedback_type
        ''')
        feedback_counts = dict(cursor.fetchall())

        # Get overall stats
        cursor.execute('SELECT COUNT(*) FROM human_feedback')
        total_feedback = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM human_feedback WHERE is_correct = 1')
        correct_predictions = cursor.fetchone()[0]

        conn.close()

        # Calculate metrics
        tp = feedback_counts.get("TP", 0)
        tn = feedback_counts.get("TN", 0)
        fp = feedback_counts.get("FP", 0)
        fn = feedback_counts.get("FN", 0)

        accuracy = (tp + tn) / total_feedback if total_feedback > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "total_feedback": total_feedback,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": total_feedback - correct_predictions,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def get_training_data(self) -> List[Dict]:
        """Get all feedback data formatted for model training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_response, human_label, ml_prediction, ml_confidence,
                   is_correct, feedback_type, timestamp
            FROM human_feedback
            ORDER BY timestamp ASC
        ''')

        results = cursor.fetchall()
        conn.close()

        training_data = []
        for row in results:
            training_data.append({
                "text": row[0],
                "true_label": row[1],
                "predicted_label": row[2],
                "confidence": row[3],
                "is_correct": bool(row[4]),
                "feedback_type": row[5],
                "timestamp": row[6]
            })

        return training_data

    def _update_performance_metrics(self):
        """Update the performance metrics table"""
        summary = self.get_feedback_summary()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metric_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO model_performance
            (id, timestamp, total_feedback, correct_predictions, incorrect_predictions,
             true_positives, true_negatives, false_positives, false_negatives,
             accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric_id, summary["timestamp"], summary["total_feedback"],
            summary["correct_predictions"], summary["incorrect_predictions"],
            summary["true_positives"], summary["true_negatives"],
            summary["false_positives"], summary["false_negatives"],
            summary["accuracy"], summary["precision"], summary["recall"], summary["f1_score"]
        ))

        conn.commit()
        conn.close()

    def _backup_to_json(self):
        """Backup all data to JSON file"""
        training_data = self.get_training_data()
        summary = self.get_feedback_summary()

        backup_data = {
            "metadata": {
                "backup_timestamp": datetime.datetime.now().isoformat(),
                "total_records": len(training_data)
            },
            "summary": summary,
            "training_data": training_data
        }

        with open(self.json_backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)

    def export_for_retraining(self, format: str = "json") -> str:
        """Export data in format suitable for model retraining"""
        training_data = self.get_training_data()

        if format == "json":
            filename = f"training_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2)
            return filename
        elif format == "csv":
            import csv
            filename = f"training_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as f:
                if training_data:
                    writer = csv.DictWriter(f, fieldnames=training_data[0].keys())
                    writer.writeheader()
                    writer.writerows(training_data)
            return filename
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_all_feedback(self):
        """Clear all feedback data (use with caution!)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM human_feedback')
        cursor.execute('DELETE FROM model_performance')

        conn.commit()
        conn.close()

        # Also clear JSON backup
        if Path(self.json_backup_path).exists():
            Path(self.json_backup_path).unlink()

# API endpoint integration
if __name__ == "__main__":
    # Example usage and testing
    storage = HumanFeedbackStorage()

    # Example feedback
    feedback_id = storage.store_feedback(
        scenario_id=1,
        user_response="What do you think happened to your tower?",
        ml_prediction="OEQ",
        ml_confidence=0.85,
        human_label="OEQ",
        session_id="demo_session_1"
    )

    print(f"Stored feedback with ID: {feedback_id}")
    print(f"Summary: {storage.get_feedback_summary()}")