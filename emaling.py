import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_bulk_email(sender_email: str, sender_password: str, recipient_list: str, subject: str, message: str, attachment_paths: list):
    """
    Send an email with attachments to multiple recipients.

    Args:
        sender_email (str): The email address of the sender.
        sender_password (str): The password or app-specific password for the sender's email account.
        recipient_list (str): A comma-separated list of recipient email addresses.
        subject (str): The subject line of the email.
        message (str): The body of the email message.
        attachment_paths (list): A list of file paths for attachments to be included in the email.

    Returns:
        None

    Raises:
        Exception: If an error occurs during the email sending process.
    """
    
    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = subject
    msg['To'] = sender_email 
    msg['Bcc'] = recipient_list

    # Add body to email
    msg.attach(MIMEText(message, 'plain', 'utf-8'))
    
    # Add attachment if provided
    for file in attachment_paths:
        with open(file, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read(), charset='utf-8')
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(file)}")
            msg.attach(part)
    try:
        # Create SMTP session
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            # Login to the server
            server.login(sender_email, sender_password)
            # Send email to all recipients at once
            server.send_message(msg)
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example:

if __name__ == '__main__':
    sender_email = "email@gmail.com"
    sender_password = "password"
    recipient_list = "email_1@gmail.com, email_2@yandex.ru"
    subject = "Test Subject"
    message = "This is a test message."
    attachment_paths = ["message.txt"]  # Optional
    send_bulk_email(sender_email, sender_password, recipient_list, subject, message, "message.txt")

