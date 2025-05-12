import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

import asyncio

load_dotenv()
TOKEN = os.environ.get("DISCORD_TOKEN")
TEST_GUILD = discord.Object(id=1364576079729266749)

bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())


async def load():
    """Load cogs as extensions."""
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), "cogs")):
        if filename.endswith(".py"):
            await bot.load_extension(f"detoxy.bot.cogs.{filename[:-3]}")


async def main():
    async with bot:
        await load()
        await bot.start(TOKEN)


asyncio.run(main())
