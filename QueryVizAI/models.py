from django.db import models

class UserData(models.Model):
    id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    gender = models.CharField(max_length=50)
    ip_address = models.GenericIPAddressField()

    def __str__(self):
        return f"{self.first_name} {self.last_name}"
