import discord
from bot import messages
from bot.logger import setup_logger
from discord.ext import commands

logger = setup_logger("listener")


class Listener(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._last_member = None

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        logger.info("Listener cog is ready")

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member) -> None:
        channel = member.guild.system_channel
        if channel is not None:
            await channel.send(messages.WELCOME_MESSAGE.format(member=member.mention))

    @commands.command()
    async def hello(self, ctx: commands.Context, member: discord.Member) -> None:
        member = member or ctx.author
        if self._last_member is None or self._last_member.id != member.id:
            await ctx.send(f"Hey {member.name}!")
        else:
            await ctx.send(f"Hey {member.name}... This feels familiar.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Listener(bot))
