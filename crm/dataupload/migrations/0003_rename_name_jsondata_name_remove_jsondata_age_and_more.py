# Generated by Django 5.1.2 on 2024-10-25 18:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataupload', '0002_remove_jsondata_created_at_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='jsondata',
            old_name='name',
            new_name='Name',
        ),
        migrations.RemoveField(
            model_name='jsondata',
            name='age',
        ),
        migrations.AddField(
            model_name='jsondata',
            name='City',
            field=models.CharField(blank=True, default='Not specified', max_length=100),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='CustomerID',
            field=models.CharField(default='unknown', max_length=50, unique=True),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Feedback',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Gender',
            field=models.CharField(blank=True, default='Not specified', max_length=10),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Income_Level',
            field=models.CharField(blank=True, default='Not specified', max_length=50),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Interests',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Last_Purchase_Date',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Purchase_History',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Region',
            field=models.CharField(blank=True, default='Not specified', max_length=100),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='Age',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
