import os
import sqlite3

from config import Config


def init_database() -> None:
    database_dir = Config.database_dir
    os.makedirs(database_dir, exist_ok=True)

    db_path = database_dir / "warnings.db"
    if not db_path.exists():
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_warnings (
            "user_id" INTEGER,
            "guild_id" INTEGER,
            "warnings" INTEGER,
            PRIMARY KEY ("user_id", "guild_id")
        )
        """)
        connection.commit()
        connection.close()
        print(f"Created database file at {db_path}")
    else:
        print(f"Database file already exists at {db_path}")


def get_warnings(user_id: int, guild_id: int) -> int:
    connection = sqlite3.connect(Config.database_dir / "warnings.db")
    cursor = connection.cursor()
    cursor.execute(
        """
    SELECT warnings FROM user_warnings WHERE user_id = ? AND guild_id = ?
    """,
        (user_id, guild_id),
    )
    warnings = cursor.fetchone()
    connection.close()

    if warnings:
        return warnings
    return 0


def increase_and_get_warnings(user_id: int, guild_id: int) -> int:
    connection = sqlite3.connect(Config.database_dir / "warnings.db")
    cursor = connection.cursor()
    cursor.execute(
        """
    SELECT warnings FROM user_warnings WHERE user_id = ? AND guild_id = ?
    """,
        (user_id, guild_id),
    )
    result = cursor.fetchone()

    if result:
        warnings = result[0] + 1
        cursor.execute(
            """
        UPDATE user_warnings SET warnings = ? WHERE user_id = ? AND guild_id = ?
        """,
            (warnings, user_id, guild_id),
        )
    else:
        warnings = 1
        cursor.execute(
            """
        INSERT INTO user_warnings (user_id, guild_id, warnings) VALUES (?, ?, ?)
        """,
            (user_id, guild_id, warnings),
        )

    connection.commit()
    connection.close()

    return warnings


if __name__ == "__main__":
    init_database()
