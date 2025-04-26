import os

import discord
from discord import Message
from discord.ext import commands
from dotenv import load_dotenv

# Load enviroment variables from .env file
load_dotenv()


class MyClient(commands.Bot):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message: Message):
        # Log message to console
        print(f"Message from {message.author}: {message.content}")

        # Ignore messages sent by our bot
        if message.author == self.user:
            return

        if message.content.startswith("!hello"):
            await message.channel.send(f"Hello @{message.author}!")

        if "fuck" not in message.content:
            return

        await message.delete()  # Delete message
        await message.channel.send("Someone sent a toxic message!")  # Alert channel
        await message.author.create_dm()  # Open DM with author
        await message.author.dm_channel.send("This is your warning!")


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(command_prefix="!", intents=intents)
    client.run(str(os.environ.get("DISCORD_TOKEN")))
