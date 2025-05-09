"""
This module contains all message templates for the Discord bot.
These messages are designed to maintain a positive and healthy community environment.
"""

# Welcome and User Interaction Messages
HELLO_MESSAGE = "Hey {mention}! 👋"
WELCOME_MESSAGE = "👋 Welcome {member}! We're excited to have you join our positive community. Feel free to introduce yourself!"
WELCOME_BACK = "Welcome back {member}! Great to see you again in our community."
USER_LEAVE = "Take care {member}! We hope to see you again soon."

# Community Guidelines and Reminders
COMMUNITY_GUIDELINES = "Remember our community values: Be kind, respectful, and supportive of others. Together we create a positive space for everyone!"
POSITIVE_REMINDER = "💫 Let's keep our interactions positive and constructive. Every message contributes to our community's atmosphere!"
HELPFUL_TIP = "💡 Tip: Before sending a message, consider how it might impact others. Kindness goes a long way!"

# Warning and Moderation Messages
TOXIC_CHANNEL_ALERT = "Hey {mention}, I noticed your message might be harmful to others. Let's keep our community positive and supportive."
TOXIC_DM_WARNING = "Hey @{author}, I noticed your message might be harmful to others. Let's keep our community positive and supportive. This is a friendly reminder to be mindful of our community guidelines."
TOXIC_DM_FOLLOW_UP = "Remember, our goal is to create a safe and welcoming space for everyone. If you need help understanding our guidelines, feel free to ask!"
REPEAT_OFFENSE = "⚠️ @{author}, this is your second warning. Continued harmful behavior may result in temporary restrictions."
TIMEOUT_MESSAGE = "⏰ @{author}, you have been timed out for 5 minutes due to repeated violations. Please take this time to review our community guidelines."

# Positive Reinforcement
POSITIVE_MESSAGE_ACK = (
    "🌟 That's a great contribution to our community! Keep spreading positivity!"
)
COMMUNITY_HIGHLIGHT = "✨ @{author} just shared something amazing! This is exactly the kind of positive interaction we love to see!"

# Bot Status and System Messages
BOT_READY = (
    "🤖 Detoxy is online and ready to help maintain a positive community environment!"
)
MESSAGE_LOG = "📝 Message from {author}: {content}"
SYSTEM_UPDATE = "🔄 System update: {update_message}"

# Error and Technical Messages
ERROR_MESSAGE = (
    "❌ Oops! Something went wrong. Our team has been notified and will fix it soon."
)
MAINTENANCE_MODE = "🔧 The bot is currently under maintenance. We'll be back shortly with improvements!"

# Community Support
HELP_COMMAND = "Need help? Here are some commands:\n!guidelines - View community guidelines\n!report - Report inappropriate behavior\n!help - Show this help message"
REPORT_CONFIRMATION = (
    "✅ Thank you for your report. Our moderation team will review it promptly."
)
SUGGESTION_ACK = "💡 Thanks for your suggestion @{author}! We value community input for making our space better."

# Special Events and Celebrations
COMMUNITY_MILESTONE = "🎉 We've reached {milestone}! Thank you all for being part of our amazing community!"
MEMBER_ANNIVERSARY = "🎊 Happy {years} year anniversary in our community, @{author}! Thank you for being a valued member!"
