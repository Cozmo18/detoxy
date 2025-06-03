import os
import sqlite3

from detoxy.bot.config import Config


def create_warnings_table():
    # Ensure database directory exists
    os.makedirs(Config.database_dir, exist_ok=True)

    connection = sqlite3.connect(Config.database_dir / "user_warnings.db")
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


def increase_and_get_warnings(user_id: int, guild_id: int) -> int:
    connection = sqlite3.connect(Config.database_dir / "user_warnings.db")
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
