import requests

from loguru import logger

 
url = "http://127.0.0.1:8010/Card_segment/upload"

def test_api(url):
    try:
        logger.info("---Card Segmentation---")
        header = {"accept": "application/json", "Content-Type": "multipart/form-data"}

        logger.info("Test single image")
        logger.info(url)
        # Test single images
        file_name_0 = "D:/AI/CV/FTech/Lib_test/test_lib_img/test_img_1.jpg"
        files={"files": ("%s" % file_name_0, open(file_name_0, "rb"), "image/")}
        # "files" must to duplicate to Upload multifile FastAPI
        r = requests.post(
            url,
            files={"files": ("%s" % file_name_0, open(file_name_0, "rb"), "image/")},
        )
        print(files)
        print("Status code:", r.status_code)
        # print(r.text)
        logger.info(f"Segmentation Result: {r.json()}")

        # Test multiple images
        logger.info("Test multiple images")
        # Upload file variables have form (file_name, content_type)
        # Examples for test upload file, we can try to wrong or true file
        # and see results.
        # file have wrong content_type
        # file_name_1 = ("tests/test_api.txt", "image/")
        # file have true content_type
        # file_name_1 = ("tests/test_api.txt", "text/")

        # true files
        file_name_1 = ("D:/AI/CV/FTech/Lib_test/test_lib_img/test_img_1.jpg", "image/")
        file_name_2 = ("D:/AI/CV/FTech/Lib_test/test_lib_img/test_img_2.jpg", "image/")
        file_name_3 = ("D:/AI/CV/FTech/Lib_test/test_lib_img/test_img_3.jpg", "image/")

        #        r = requests.post(url, files = [("files", ("%s"%file_name_0, open(file_name_0, "rb"), "image/")),
        #                                        ("files", ("%s"%file_name_1, open(file_name_1, "rb"), "image/")),
        #                                        ("files", ("%s"%file_name_2, open(file_name_2, "rb"), "image/")),
        #                                        ("files", ("%s"%file_name_3, open(file_name_3, "rb"), "image/"))])

        List_test_file = [
            file_name_1,
            file_name_2,
            file_name_3,
        ]
        r = requests.post(
            url,
            files=[
                (
                    "files",
                    (
                        "%s" % file_name[0],
                        open(file_name[0], "rb"),
                        "%s" % file_name[1],
                    ),
                )
                for file_name in List_test_file
            ],
        )
        if r.status_code == 500:
            logger.info("Error content_type: Test files contain file that isn't image")
        elif r.status_code == 200:
            print(r.status_code)
            logger.info(f" Segmentation Result: {r.json()}")
            # print(r.json())

        # print("Right")
    except Exception as e:
        print(requests)
        logger.exception(e)
        # pass


def main():

    test_api(url)


if __name__ == "__main__":
    main()
