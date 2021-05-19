import urllib.request
urllib.request.urlretrieve(
    "https://content-na.drive.amazonaws.com/cdproxy/batchLink/29749zQQWaHq7LsGM7DUQiWjReGoWUlGZ32hAnQBXXw/content",
    "/gpfs1/scratch/ckchan666/xtreme/AmazonPhotos.zip")
import xtreme_ds
for task in xtreme_ds.TASK:
    try:
        set_name, subset_name, split = xtreme_ds.TASK[task]['train']
        xtreme_ds.get_dataset(set_name, subset_name)[split]
    except:
        pass
xtreme_ds.summary()