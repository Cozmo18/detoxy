import discord
from discord.ext import commands
import aiohttp  # Replace requests with aiohttp for async HTTP requests

from detoxy.bot import messages
from detoxy.bot.config import Config


class Moderation(commands.Cog):
    def __init__(self, bot, threshold: float = Config.threshold):
        self.bot = bot
        self.threshold = threshold
        self.user_warnings = {}

    @commands.Cog.listener()
    async def on_ready(self):
        print(f"{__name__} is ready!")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user:
            return
        else:
            print(
                messages.MESSAGE_LOG.format(
                    author=message.author, content=message.content
                )
            )

        toxic_labels = await self.predict_toxicity(message)
        if len(toxic_labels) > 0:
            await self.handle_toxic_message(message, toxic_labels)
        else:
            pass

    async def handle_toxic_message(
        self, message: discord.Message, toxic_labels: list[str]
    ):
        """Handle toxic message detection"""
        print(f"Toxic message detected ({toxic_labels}): {message.content}")
        await message.delete()
        await message.channel.send(
            messages.TOXIC_CHANNEL_ALERT.format(mention=message.author.mention)
        )

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
            elif warning_count == 2:
                await dm_channel.send(
                    messages.REPEAT_OFFENSE.format(author=message.author.mention)
                )
            else:
                # Issue a 5-minute timeout
                # await message.author.timeout(timedelta(minutes=1), reason="Multiple toxic messages")
                await dm_channel.send(
                    messages.TIMEOUT_MESSAGE.format(author=message.author.mention)
                )
                # Reset warning count after timeout
                # self.user_warnings[user_id] = 0

        except discord.Forbidden:
            print(f"Could not send DM to user {message.author}")

    async def predict_toxicity(self, message: discord.Message) -> list[str]:
        data = {"input": message.content}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="http://127.0.0.1:8000/predict", json=data
            ) as response:
                result = await response.json()
                toxic_labels = [
                    label for label, prob in result.items() if prob >= self.threshold
                ]
                return toxic_labels


async def setup(bot):
    await bot.add_cog(Moderation(bot))
