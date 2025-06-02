from datetime import timedelta

import aiohttp
import discord
from discord.ext import commands

from detoxy.bot import messages
from detoxy.bot.config import Config
from detoxy.bot.logger import setup_logger

logger = setup_logger("moderation", Config.log_dir)


class Moderation(commands.Cog):
    def __init__(self, bot, threshold: float = Config.threshold):
        self.bot = bot
        self.threshold = threshold
        self.user_warnings = {}

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
        logger.warning(f"Toxic message detected: {message.content}")
         
        await message.delete()
        
        user_id = message.author.id
        if user_id not in self.user_warnings:
            self.user_warnings[user_id] = 0

        self.user_warnings[user_id] += 1
        warning_count = self.user_warnings[user_id]

        try:
            dm_channel = await message.author.create_dm()
            if warning_count == 1:
                await dm_channel.send(
                    messages.TOXIC_DM_WARNING.format(author=message.author.mention)
                )
                logger.info(f"Sent first warning DM to user {message.author}")
            elif warning_count == 2:
                await dm_channel.send(
                    messages.REPEAT_OFFENSE.format(author=message.author.mention)
                )
                logger.info(f"Sent second warning DM to user {message.author}")
            else:
                # Issue a 5-minute timeout
                await message.author.timeout(
                    timedelta(minutes=1), reason="Multiple toxic messages"
                )
                await dm_channel.send(
                    messages.TIMEOUT_MESSAGE.format(author=message.author.mention)
                )
                logger.info(
                    f"Issued a 1 minute timeout to {message.author} and sent timeout notification DM"
                )
                # Reset warning count after timeout
                # self.user_warnings[user_id] = 0

        except discord.Forbidden as e:
            logger.error(f"Failed to send DM to user {message.author}: {str(e)}")

    async def predict_toxicity(self, message: discord.Message) -> dict:
        data = {"input": message.content}

        async with aiohttp.ClientSession() as session:
            async with session.post(url=Config.predict_url, json=data) as response:
                result = await response.json()
                return result


async def setup(bot):
    await bot.add_cog(Moderation(bot))
