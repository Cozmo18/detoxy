from discord.ext import commands

from database import get_warnings
from logger import setup_logger

logger = setup_logger("admin")


class Admin(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info("Admin cog is ready")

    @commands.command()
    async def get_warnings(self, ctx: commands.Context, user: str) -> int:
        guild_id = ctx.guild.id
        member = ctx.guild.get_member_named(user)
        user_id = member.id
        warnings = get_warnings(user_id, guild_id)
        await ctx.send(f"{member} has {warnings} warnings.")


async def setup(bot):
    await bot.add_cog(Admin(bot))
