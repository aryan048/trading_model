from email.message import EmailMessage
import os
import smtplib

def send_sms_notification(message):
    phone_number = "+1" + os.getenv('phone_number')  # Your iPhone number

    os.system(f"osascript -e 'tell application \"Messages\" to send \"{message}\" to buddy \"{phone_number}\"'")

if __name__ == "__main__":
    send_sms_notification("Test")