import asyncio
from pathlib import Path

import discord
from discord.ext import commands

from bot.config import DISCORD_TOKEN
from bot.logger import setup_logger

TEST_GUILD = discord.Object(id=1380173035952410624)

logger = setup_logger("bot")
bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())


async def load() -> None:
    """Load cogs as extensions."""
    for filename in (Path(__file__).parent / "cogs").iterdir():
        if filename.endswith(".py"):
            try:
                await bot.load_extension(f"bot.cogs.{filename[:-3]}")
                logger.info("Loaded extension: %s", filename[:-3])
            except Exception:
                logger.exception("Failed to load extension %s", filename[:-3])


async def main() -> None:
    async with bot:
        await load()
        await bot.start(DISCORD_TOKEN)
        logger.info("Detoxy bot started!")


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
