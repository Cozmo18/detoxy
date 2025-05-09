import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
from detoxy.bot import messages

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


# class DetoxyBot(discord.Client):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tree = app_commands.CommandTree(self)
#         self.user_warnings = {}
#         self.positive_message_count = 0

#     async def setup_hook(self):
#         """This is called when the bot is starting up"""
#         self.tree.copy_global_to(guild=TEST_GUILD)
#         await self.tree.sync(guild=TEST_GUILD)

#     @commands.command(name="help")
#     async def help_command(self, ctx):
#         """Display help message with available commands"""
#         await ctx.send(messages.HELP_COMMAND)

#     async def on_ready(self):
#         print(messages.BOT_READY.format(user=self.user))
#         # await self.change_presence(activity=discord.Game(name="!help for commands"))

#     async def on_member_join(self, member):
#         """Handle new member joins"""
#         channel = member.guild.system_channel
#         if channel:
#             await channel.send(messages.HELLO_MESSAGE.format(author=member.mention))

#     async def on_member_remove(self, member):
#         """Handle member leaves"""
#         channel = member.guild.system_channel
#         if channel:
#             await channel.send(messages.USER_LEAVE.format(author=member.mention))

#     async def on_message(self, message: Message):
#         print(messages.MESSAGE_LOG.format(author=message.author, content=message.content))

#         if message.author == self.user:
#             return

#         is_toxic, toxic_labels = predict_toxicity(message)

#         if is_toxic:
#             await self.handle_toxic_message(message, toxic_labels)
#         else:
#             # Randomly send positive reinforcement (5% chance)
#             if message.content and len(message.content) > 20 and random.random() < 0.05:
#                 await message.add_reaction("🌟")
#                 self.positive_message_count += 1

#                 # Every 10 positive messages, send a community highlight
#                 if self.positive_message_count % 10 == 0:
#                     await message.channel.send(messages.COMMUNITY_HIGHLIGHT.format(author=message.author.mention))

#         await self.process_commands(message)

#     async def handle_toxic_message(self, message: Message, toxic_labels: dict):
#         """Handle toxic message detection"""
#         print(f"Toxic message detected ({toxic_labels}): {message.content}")

#         await message.delete()
#         await message.channel.send(messages.TOXIC_CHANNEL_ALERT.format(mention=message.author.mention))

#         user_id = message.author.id
#         if user_id not in self.user_warnings:
#             self.user_warnings[user_id] = 0

#         self.user_warnings[user_id] += 1
#         warning_count = self.user_warnings[user_id]

#         try:
#             dm_channel = await message.author.create_dm()

#             if warning_count == 1:
#                 await dm_channel.send(messages.TOXIC_DM_WARNING.format(author=message.author.mention))
#             elif warning_count == 2:
#                 await dm_channel.send(messages.REPEAT_OFFENSE.format(author=message.author.mention))
#             else:
#                 # Issue a 5-minute timeout
#                 # await message.author.timeout(timedelta(minutes=1), reason="Multiple toxic messages")
#                 await dm_channel.send(messages.TIMEOUT_MESSAGE.format(author=message.author.mention))
#                 # Reset warning count after timeout
#                 # self.user_warnings[user_id] = 0

#         except discord.Forbidden:
#             print(f"Could not send DM to user {message.author}")
