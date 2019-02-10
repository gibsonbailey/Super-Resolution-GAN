import requests
import getpass
import sys

LOGIN_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/login/"
FILE_URLS = [
    "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz",
    "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz",
    "http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/train_list.txt",
    "http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_list.txt",
    "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/identity_meta.csv",
    "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/class_overlap_vgg1_2.txt",
    "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_posetemp_imglist.txt",
    "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_agetemp_imglist.txt",
    "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/bb_landmark.tar.gz"
]
META_KEYWORD = 'meta'

def get_filenames(urls):
    filenames = []
    for url in urls:
        filename = url.split('/')[-1]
        if '=' in filename:
            filename = filename.split('=')[-1]
        filenames.append(filename)
    return filenames

def get_credentials():
    print('Please enter your VGG Face 2 credentials:')
    user_string = input('    User: ')
    password_string = getpass.getpass(prompt='    Password: ')

    credentials =  {
        'username': user_string,
        'password': password_string
    }

    return credentials

def get_file_choices(filenames, file_urls):
    print("Which file would you like to download?")
    for i, filename in enumerate(filenames):
        print(f"    [{i}]: {filename}")
    print(f"    [{len(filenames)}]: All files")
    print(f"    [{META_KEYWORD}]: Meta files")

    choice = None
    while (choice == None):
        choice = input(">>> ")
        if choice == META_KEYWORD:
            continue
        else: 
            try:
                choice = int(choice)
                if choice < 0 or choice > len(filenames):
                    print(f"Invalid choice: {choice}. Please try again.")
            except ValueError:
                print(f"Unrecognized input. Please try again.")
                choice = None

    if choice == META_KEYWORD:
        choices = []
        for filename, file_url in zip(filenames, file_urls):
            if META_KEYWORD in file_url:
                choices.append((filename, file_url))
    elif choice == len(filenames):
        choices = []
        for filename, file_url in zip(filenames, file_urls):
            choices.append((filename, file_url))
    else:
        choices = [(filenames[choice], file_urls[choice])]
    return choices

if __name__ == "__main__":
    session = requests.session()
    r = session.get(LOGIN_URL)

    if 'csrftoken' in session.cookies:
        csrftoken = session.cookies['csrftoken']
    elif 'csrf' in session.cookies:
        csrftoken = session.cookies['csrf']
    else:
        raise ValueError("Unable to locate CSRF token. "
                         "Is the login url correct?")

    credentials = get_credentials()
    credentials['csrfmiddlewaretoken'] = csrftoken

    r = session.post(LOGIN_URL, data=credentials)

    # Ask users which files to download
    filenames = get_filenames(FILE_URLS)
    choices = get_file_choices(filenames, FILE_URLS)

    for filename, file_url in choices:
        with open(filename, "wb") as f:
            print(f"Downloading file: `{filename}`")
            r = session.get(file_url, data=credentials, stream=True)

            total_bytes = r.headers.get('content-length')
            encoding = r.headers.get('content-encoding')
            zipped = (encoding is not None and 'zip' in encoding)
            if total_bytes is None:
                print(f"    Size: (not provided)")
            elif zipped:
                print(f"    Size: {total_bytes} bytes (compressed)")
            else:
                print(f"    Size: {total_bytes} bytes")

            try:
                if total_bytes is None or zipped:
                    # Display number of megabytes downloaded so far.
                    bytes_written = 0
                    for data in r.iter_content(chunk_size=4096):
                        f.write(data)
                        bytes_written += len(data)
                        MiB = bytes_written / (1024 * 1024)
                        sys.stdout.write(f"\r{MiB:0.2f} MiB downloaded...")
                        sys.stdout.flush()
                else:
                    # Display progress bar
                    total_bytes = int(total_bytes)
                    bytes_written = 0
                    
                    bar_length = 50
                    write_str = "\r[{}]".format("{:" + str(bar_length) + "s}")

                    for data in r.iter_content(chunk_size=4096): # Chunk size?
                        f.write(data)
                        bytes_written += len(data)

                        fraction_complete = bytes_written / total_bytes
                        num_bars = int(fraction_complete * bar_length)
                        percent = int(fraction_complete * 100)
                        
                        sys.stdout.write(write_str.format('=' * num_bars))
                        sys.stdout.write(" {}%".format(percent))    
                        sys.stdout.flush()
            except KeyboardInterrupt:
                print("\nKeyboard Interrupt: skipping this file.")
        print()
    print("Done.")