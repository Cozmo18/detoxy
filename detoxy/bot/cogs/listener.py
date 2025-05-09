from discord.ext import commands
import discord
from detoxy.bot import messages


class Listener(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self._last_member = None

    @commands.Cog.listener()
    async def on_ready(self):
        print(f"{__name__} is ready!")

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        channel = member.guild.system_channel
        if channel is not None:
            await channel.send(messages.WELCOME_MESSAGE.format(member=member.mention))

    @commands.command()
    async def hello(self, ctx: commands.Context, *, member: discord.Member = None):
        member = member or ctx.author
        if self._last_member is None or self._last_member.id != member.id:
            await ctx.send(f"Hey {member.name}!")
        else:
            await ctx.send(f"Hey {member.name}... This feels familiar.")
        self._last_member = member


async def setup(bot):
    await bot.add_cog(Listener(bot))
