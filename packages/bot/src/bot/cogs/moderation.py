from datetime import timedelta

import aiohttp
import discord
from bot import messages
from bot.config import API_URL, THRESHOLD
from bot.logger import setup_logger
from discord.ext import commands

logger = setup_logger("moderation")


class Moderation(commands.Cog):
    def __init__(self, bot, threshold: float = THRESHOLD) -> None:
        self.bot = bot
        self.threshold = threshold
        self.user_warnings = {}

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        logger.info("Moderation cog is ready")

    @commands.Cog.listener()
    @commands.bot_has_guild_permissions(administrator=True)
    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.bot.user:
            return

        try:
            toxicity = await self.predict_toxicity(message)
            if any(v > self.threshold for v in toxicity.values()):
                await self.handle_toxic_message(message)
        except Exception as e:
            logger.exception(f"Error processing message: {e}")

    async def handle_toxic_message(self, message: discord.Message) -> None:
        """Handle toxic message detection"""
        await self._delete_message(message)
        warnings = await self._update_warnings(message)
        await self._handle_warning_level(message, warnings)

    async def _delete_message(self, message: discord.Message) -> None:
        """Delete the toxic message and log the action."""
        try:
            await message.delete()
            logger.info(f"Toxic message detected and deleted: {message.content}")
        except discord.Forbidden:
            logger.exception("Bot lacks permissions to delete messages")
        except discord.NotFound:
            logger.exception("Message was already deleted")
        except Exception as e:
            logger.exception(f"Failed to delete toxic message: {e}")

    async def _update_warnings(self, message: discord.Message) -> int:
        """Update warning count for the user and return current warning count."""
        try:
            user_id = message.author.id
            self.user_warnings[user_id] = self.user_warnings.get(user_id, 0) + 1
            warnings = self.user_warnings[user_id]
            logger.debug(f"User {user_id} has {warnings} warnings")
            return warnings
        except Exception as e:
            logger.exception(f"Error updating warnings: {e}")
            return 0

    async def _handle_warning_level(
        self, message: discord.Message, warnings: int
    ) -> None:
        """Handle different warning levels and take appropriate action."""
        if warnings == 1:
            await self._send_first_warning(message)
        elif warnings == 2:
            await self._send_second_warning(message)
        else:
            await self._handle_timeout(message)

    async def _send_first_warning(self, message: discord.Message) -> None:
        """Send first warning DM to the user."""
        try:
            await message.author.send(
                messages.TOXIC_DM_WARNING.format(author=message.author.mention)
            )
            logger.info(f"Sent first warning DM to user {message.author}")
        except discord.Forbidden:
            logger.warning(
                f"Could not send first warning DM to {message.author} - "
                "Bot does not have permissions."
            )
        except Exception as e:
            logger.warning(f"Could not send first warning DM: {e}")

    async def _send_second_warning(self, message: discord.Message) -> None:
        """Send second warning DM to the user."""
        try:
            await message.author.send(
                messages.REPEAT_OFFENSE.format(author=message.author.mention)
            )
            logger.info(f"Sent second warning DM to user {message.author}")
        except discord.Forbidden:
            logger.warning(
                f"Could not send second warning DM to {message.author} - DMs closed"
            )
        except Exception as e:
            logger.warning(f"Could not send second warning DM: {e}")

    async def _handle_timeout(self, message: discord.Message) -> None:
        """Handle timeout for users with multiple violations."""
        try:
            await message.author.timeout(
                timedelta(minutes=5), reason="Multiple toxic messages"
            )
            logger.info(f"Timed out user {message.author} for 5 minutes")
        except discord.Forbidden:
            logger.warning(
                f"Could not timeout user {message.author} - insufficient permissions"
            )
        except Exception as e:
            logger.warning(f"Could not issue timeout: {e}")

        try:
            await message.author.send(
                messages.TIMEOUT_MESSAGE.format(author=message.author.mention)
            )
            logger.info(f"Sent timeout notification to user {message.author}")
        except discord.Forbidden:
            logger.warning(
                f"Could not send timeout message to {message.author} - DMs closed"
            )
        except Exception as e:
            logger.warning(f"Could not send timeout message to user: {e}")

    async def predict_toxicity(self, message: discord.Message) -> dict:
        try:
            data = {"input": message.content}
            async with (
                aiohttp.ClientSession() as session,
                session.post(url=API_URL, json=data) as response,
            ):
                if response.status != 200:
                    logger.error(
                        f"Prediction service returned status {response.status}"
                    )
                    return {}
                result = await response.json()
                logger.debug(f"Toxicity prediction result: {result}")
                return result
        except aiohttp.ClientError as e:
            logger.exception(f"Network error while predicting toxicity: {e}")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error while predicting toxicity: {e}")
            return {}


async def setup(bot) -> None:
    await bot.add_cog(Moderation(bot))
