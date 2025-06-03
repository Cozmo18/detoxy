from datetime import timedelta

import aiohttp
import discord
from discord.ext import commands

from detoxy.bot import messages
from detoxy.bot.config import Config
from detoxy.bot.database import create_warnings_table, increase_and_get_warnings
from detoxy.bot.logger import setup_logger

logger = setup_logger("moderation", Config.log_dir)


class Moderation(commands.Cog):
    def __init__(self, bot, threshold: float = Config.threshold):
        self.bot = bot
        self.threshold = threshold

        create_warnings_table()

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info("Moderation cog is ready")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user:
            return

        toxicity = await self.predict_toxicity(message)
        if any(v > self.threshold for v in toxicity.values()):
            await self.handle_toxic_message(message)

    async def handle_toxic_message(self, message: discord.Message):
        """Handle toxic message detection"""
        await message.delete()
        logger.warning(f"Toxic message detected and deleted: {message.content}")

        guild_id = message.guild.id
        user_id = message.author.id
        warnings = increase_and_get_warnings(user_id, guild_id)

        if warnings == 1:
            await message.author.send(
                messages.TOXIC_DM_WARNING.format(author=message.author.mention)
            )
            logger.info(f"Sent first warning DM to user {message.author}")
        elif warnings == 2:
            await message.author.send(
                messages.REPEAT_OFFENSE.format(author=message.author.mention)
            )
            logger.info(f"Sent second warning DM to user {message.author}")
        else:
            await message.author.timeout(
                timedelta(minutes=5), reason="Multiple toxic messages"
            )
            await message.author.send(
                messages.TIMEOUT_MESSAGE.format(author=message.author.mention)
            )
            logger.info(
                f"Issued a 5 minute timeout to {message.author} and sent timeout notification DM"
            )

    async def predict_toxicity(self, message: discord.Message) -> dict:
        data = {"input": message.content}
        async with aiohttp.ClientSession() as session:
            async with session.post(url=Config.predict_url, json=data) as response:
                result = await response.json()
                return result


async def setup(bot):
    await bot.add_cog(Moderation(bot))
