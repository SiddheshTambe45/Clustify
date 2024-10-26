# Generated by Django 5.1.2 on 2024-10-25 16:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataupload', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='jsondata',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='jsondata',
            name='json_content',
        ),
        migrations.AddField(
            model_name='jsondata',
            name='age',
            field=models.IntegerField(default=None),
        ),
        migrations.AddField(
            model_name='jsondata',
            name='name',
            field=models.CharField(default='unknown', max_length=100),
        ),
    ]
