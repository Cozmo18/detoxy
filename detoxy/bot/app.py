import os
import random

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
    TOXIC_DM_FOLLOW_UP,
    WELCOME_BACK,
    USER_LEAVE,
    COMMUNITY_GUIDELINES,
    POSITIVE_REMINDER,
    HELPFUL_TIP,
    POSITIVE_MESSAGE_ACK,
    COMMUNITY_HIGHLIGHT,
    HELP_COMMAND,
    REPORT_CONFIRMATION,
    SUGGESTION_ACK,
    ERROR_MESSAGE,
)

from detoxy.ml.predictor import predict_toxicity


# Load environment variables from .env file
load_dotenv()


class MyClient(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_warnings = {}  # Track user warnings
        self.positive_message_count = 0  # Track positive messages for highlighting

    async def on_ready(self):
        print(BOT_READY.format(user=self.user))
        # Set bot status
        await self.change_presence(activity=discord.Game(name="!help for commands"))

    async def on_member_join(self, member):
        """Handle new member joins"""
        channel = member.guild.system_channel
        if channel:
            await channel.send(HELLO_MESSAGE.format(author=member.mention))

    async def on_member_remove(self, member):
        """Handle member leaves"""
        channel = member.guild.system_channel
        if channel:
            await channel.send(USER_LEAVE.format(author=member.mention))

    async def on_message(self, message: Message):
        # Log message to console
        print(MESSAGE_LOG.format(author=message.author, content=message.content))

        # Ignore messages sent by our bot
        if message.author == self.user:
            return

        # Process commands
        await self.process_commands(message)

        # Check for toxic content
        is_toxic, toxic_labels = predict_toxicity(message)

        if is_toxic:
            await self.handle_toxic_message(message, toxic_labels)
        else:
            # Randomly send positive reinforcement (5% chance)
            if message.content and len(message.content) > 20 and random.random() < 0.05:
                await message.add_reaction("🌟")
                self.positive_message_count += 1
                
                # Every 10 positive messages, send a community highlight
                if self.positive_message_count % 10 == 0:
                    await message.channel.send(COMMUNITY_HIGHLIGHT.format(author=message.author.mention))

    async def handle_toxic_message(self, message: Message, toxic_labels: dict):
        """Handle toxic message detection"""
        print(f"Toxic message detected ({toxic_labels}): {message.content}")
        
        # Delete the toxic message
        await message.delete()
        
        # Send alert to the channel
        await message.channel.send(TOXIC_CHANNEL_ALERT)
        
        # Handle user warning
        user_id = message.author.id
        if user_id not in self.user_warnings:
            self.user_warnings[user_id] = 0
        
        self.user_warnings[user_id] += 1
        
        # Send DM to user
        try:
            dm_channel = await message.author.create_dm()
            await dm_channel.send(TOXIC_DM_WARNING.format(author=message.author.mention))
            await dm_channel.send(TOXIC_DM_FOLLOW_UP)
        except discord.Forbidden:
            print(f"Could not send DM to user {message.author}")

    @commands.command(name="help")
    async def help_command(self, ctx):
        """Display help message with available commands"""
        await ctx.send(HELP_COMMAND)

    @commands.command(name="guidelines")
    async def guidelines(self, ctx):
        """Display community guidelines"""
        await ctx.send(COMMUNITY_GUIDELINES)

    @commands.command(name="report")
    async def report(self, ctx):
        """Handle user reports"""
        await ctx.send(REPORT_CONFIRMATION)

    @commands.command(name="suggest")
    async def suggest(self, ctx):
        """Handle community suggestions"""
        await ctx.send(SUGGESTION_ACK.format(author=ctx.author.mention))

    @commands.command(name="reminder")
    async def reminder(self, ctx):
        """Send a positive community reminder"""
        await ctx.send(POSITIVE_REMINDER)

    @commands.command(name="tip")
    async def tip(self, ctx):
        """Send a helpful community tip"""
        await ctx.send(HELPFUL_TIP)


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True  # Enable member intents for join/leave events
    client = MyClient(command_prefix="!", intents=intents)
    client.run(str(os.environ.get("DISCORD_TOKEN")))
