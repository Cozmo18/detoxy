from datetime import timedelta

import aiohttp
import discord
from discord.ext import commands

from detoxy.app import messages
from detoxy.app.config import Config
from detoxy.app.logger import setup_logger

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
    @commands.bot_has_guild_permissions(administrator=True)
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user:
            return

        try:
            toxicity = await self.predict_toxicity(message)
            if any(v > self.threshold for v in toxicity.values()):
                await self.handle_toxic_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def handle_toxic_message(self, message: discord.Message):
        """Handle toxic message detection"""
        try:
            await message.delete()
            logger.info(f"Toxic message detected and deleted: {message.content}")
        except Exception as e:
            logger.error(f"Failed to delete toxic message: {e}")
            return

        try:
            guild_id = message.guild.id
            user_id = message.author.id
            
            if user_id in self.user_warnings.keys():
                self.user_warnings[user_id] += 1
            else:
                self.user_warnings[user_id] = 1
                
            warnings = self.user_warnings[user_id]
            logger.debug(f"User {user_id} in guild {guild_id} has {warnings} warnings")
            
        except Exception as e:
            logger.error(f"Database error while handling warnings: {e}")
            return

        if warnings == 1:
            try:
                await message.author.send(
                    messages.TOXIC_DM_WARNING.format(author=message.author.mention)
                )
                logger.info(f"Sent first warning DM to user {message.author}")
            except Exception as e:
                logger.warning(f"Could not send first warning DM: {e}")
        elif warnings == 2:
            try:
                await message.author.send(
                    messages.REPEAT_OFFENSE.format(author=message.author.mention)
                )
                logger.info(f"Sent second warning DM to user {message.author}")
            except Exception as e:
                logger.warning(f"Could not send second warning DM: {e}")
        else:
            try:
                await message.author.timeout(
                    timedelta(minutes=5), reason="Multiple toxic messages"
                )
                logger.info(f"Timed out user {message.author} for 5 minutes")
            except Exception as e:
                logger.warning(f"Could not issue timeout: {e}")

            try:
                await message.author.send(
                    messages.TIMEOUT_MESSAGE.format(author=message.author.mention)
                )
                logger.info(f"Sent timeout notification to user {message.author}")
            except Exception as e:
                logger.warning(f"Could not send timeout message to user: {e}")

    async def predict_toxicity(self, message: discord.Message) -> dict:
        try:
            data = {"input": message.content}
            async with aiohttp.ClientSession() as session:
                async with session.post(url=Config.predict_url, json=data) as response:
                    if response.status != 200:
                        logger.error(
                            f"Prediction service returned status {response.status}"
                        )
                        return {}
                    result = await response.json()
                    logger.debug(f"Toxicity prediction result: {result}")
                    return result
        except aiohttp.ClientError as e:
            logger.error(f"Network error while predicting toxicity: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error while predicting toxicity: {e}")
            return {}


async def setup(bot):
    await bot.add_cog(Moderation(bot))
