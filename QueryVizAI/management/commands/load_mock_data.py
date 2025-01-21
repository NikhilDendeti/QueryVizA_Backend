import csv
from django.core.management.base import BaseCommand
from QueryVizAI.models import UserData

class Command(BaseCommand):
    help = "Load mock data from a CSV file into the database"

    def handle(self, *args, **kwargs):
        file_path = 'QueryVizAI/MOCK_DATA.csv'  # Ensure the correct relative path
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            data = [
                UserData(
                    id=row['id'],
                    first_name=row['first_name'],
                    last_name=row['last_name'],
                    email=row['email'],
                    gender=row['gender'],
                    ip_address=row['ip_address']
                )
                for row in reader
            ]
            UserData.objects.bulk_create(data, ignore_conflicts=True)
        self.stdout.write(self.style.SUCCESS("Data loaded successfully!"))
