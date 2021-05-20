import xtreme_ds
for task in xtreme_ds.TASK:
    try:
        set_name, subset_name, split = xtreme_ds.TASK[task]['train']
        xtreme_ds.get_dataset(set_name, subset_name)[split]
    except:
        pass
xtreme_ds.summary()