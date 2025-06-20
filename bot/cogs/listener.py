import discord
from discord.ext import commands

import messages
from logger import setup_logger

logger = setup_logger("listener")


class Listener(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self._last_member = None

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info("Listener cog is ready")

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        channel = member.guild.system_channel
        if channel is not None:
            await channel.send(messages.WELCOME_MESSAGE.format(member=member.mention))
            logger.info(f"New member joined: {member.name}")

    @commands.command()
    async def hello(self, ctx: commands.Context, *, member: discord.Member = None):
        member = member or ctx.author
        if self._last_member is None or self._last_member.id != member.id:
            await ctx.send(f"Hey {member.name}!")
        else:
            await ctx.send(f"Hey {member.name}... This feels familiar.")
        self._last_member = member
        logger.info(f"Hello command used by {member.name}")


async def setup(bot):
    await bot.add_cog(Listener(bot))
