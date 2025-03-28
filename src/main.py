import pre_processing.s3_download as s3_download
import SMS_notifier

if __name__ == "__main__":
    s3_download.download_data()
    SMS_notifier.send_sms_notification("DB finished populating")