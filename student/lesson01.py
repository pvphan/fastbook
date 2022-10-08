from time import sleep

from duckduckgo_search import ddg_images
from fastai.vision.all import *
from fastcore.all import *
from fastdownload import download_url


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')


def downloadImageExample(parentDir):
    urls = search_images('bird photos', max_images=1)
    dest = parentDir/'bird.jpg'
    download_url(urls[0], dest, show_progress=False)
    im = Image.open(dest)
    #im.to_thumb(256,256)


def main(parentDir):
    searches = 'forest','bird'
    path = Path(parentDir/'bird_or_not')

    print("Downloading images")
    for o in searches:
        dest = (path/o)
        if not dest.exists():
            dest.mkdir(exist_ok=True, parents=True)
            download_images(dest, urls=search_images(f'{o} photo'))
            download_images(dest, urls=search_images(f'{o} sun photo'))
            download_images(dest, urls=search_images(f'{o} shade photo'))

            resize_images(path/o, max_size=400, dest=path/o)

    print("Unlinking corrupt images")
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)

    print("Creating dataloader")
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')],
    ).dataloaders(path, bs=32)
    #dls.show_batch(max_n=6)

    print("Training learner")
    learner = vision_learner(dls, resnet18, metrics=error_rate)

    print("Find tuning learner")
    learner.fine_tune(3)

    print("Testing learner inference")
    imagePath = parentDir/'target.jpg'
    urls = search_images('monkey in forest photos', max_images=1)
    download_url(urls[-1], imagePath, show_progress=False)
    image = PILImage.create(imagePath)
    predictedClass, _, probs = learner.predict(image)
    print(f"This is a: {predictedClass}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")
    #image.to_thumb(256,256)


if __name__ == "__main__":
    parentDir = Path('student/images')
    main(parentDir)
