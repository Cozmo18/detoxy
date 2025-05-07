import os

import discord
from discord import Message
from discord.ext import commands
from dotenv import load_dotenv
from detoxy.bot.messages import (
    BOT_READY,
    HELLO_MESSAGE,
    MESSAGE_LOG,
    TOXIC_CHANNEL_ALERT,
    TOXIC_DM_WARNING,
)

from detoxy.ml.predictor import predict_toxicity


# Load enviroment variables from .env file
load_dotenv()


class MyClient(commands.Bot):
    async def on_ready(self):
        print(BOT_READY.format(user=self.user))

    async def on_message(self, message: Message):
        # Log message to console
        print(MESSAGE_LOG.format(author=message.author, content=message.content))

        # Ignore messages sent by our bot
        if message.author == self.user:
            return

        if message.content.startswith("!hello"):
            await message.channel.send(HELLO_MESSAGE.format(author=message.author))

        is_toxic, toxic_labels = predict_toxicity(message)

        if is_toxic:
            print(f"Toxic message detected ({toxic_labels}): {message.content}")
            await message.delete()  # Delete message
            await message.channel.send(TOXIC_CHANNEL_ALERT)
            await message.author.create_dm()  # Open DM with author
            await message.author.dm_channel.send(TOXIC_DM_WARNING)
        else:
            return


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(command_prefix="!", intents=intents)
    client.run(str(os.environ.get("DISCORD_TOKEN")))
