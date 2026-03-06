import pytest
from app.core.filesystem import FileSystemManager

def test_init_sets_root_to_tmp_path(tmp_path):
    fsm = FileSystemManager(root=tmp_path)
    assert fsm._root == tmp_path.resolve()
    assert fsm.worker_path == tmp_path.resolve()


def test_init_raises_if_root_not_dir(tmp_path):
    file = tmp_path / "notadir.txt"
    file.write_text("hello")
    with pytest.raises(NotADirectoryError):
        FileSystemManager(root=file)


def test_status_root(populated_fs):
    assert populated_fs.status() == "."


def test_status_after_cd(populated_fs):
    populated_fs.in_dir("photos")
    assert populated_fs.status() == "photos"


def test_in_dir_success(populated_fs):
    populated_fs.in_dir("photos")
    assert populated_fs.worker_path.name == "photos"


def test_in_dir_nonexistent_raises(populated_fs):
    with pytest.raises(FileNotFoundError):
        populated_fs.in_dir("nonexistent")


def test_in_dir_escape_root_raises(tmp_path):
    fsm = FileSystemManager(root=tmp_path)
    dangerous = tmp_path.parent
    with pytest.raises(PermissionError):
        fsm.in_dir(str(dangerous))


def test_in_dirs_chain(populated_fs):
    populated_fs.in_dirs(["photos"])
    assert populated_fs.status() == "photos"


def test_out_dir_success(populated_fs):
    populated_fs.in_dir("photos")
    populated_fs.out_dir()
    assert populated_fs.status() == "."


def test_out_dir_at_root_raises(populated_fs):
    with pytest.raises(PermissionError):
        populated_fs.out_dir()


def test_reset(populated_fs):
    populated_fs.in_dir("photos")
    populated_fs.reset()
    assert populated_fs.status() == "."


def test_get_all_dirs(populated_fs):
    dirs = populated_fs.get_all_dirs()
    assert set(dirs) == {"photos", "docs", "empty"}


def test_get_all_files(populated_fs):
    files = populated_fs.get_all_files()
    assert set(files) == {"photo1.jpg", "photo2.png", "doc.pdf", "text.txt"}


def test_rename_file_success(populated_fs):
    populated_fs.rename_file("photo1.jpg", "image1.jpg")
    assert "image1.jpg" in populated_fs.get_all_files()
    assert "photo1.jpg" not in populated_fs.get_all_files()


def test_rename_dir_success(populated_fs):
    populated_fs.rename_dir("photos", "images")
    assert "images" in populated_fs.get_all_dirs()
    assert "photos" not in populated_fs.get_all_dirs()


def test_rename_file_to_existing_raises(populated_fs):
    with pytest.raises(FileExistsError):
        populated_fs.rename_file("photo1.jpg", "doc.pdf")


def test_rename_invalid_name_raises(populated_fs):
    with pytest.raises(ValueError):
        populated_fs.rename_file("photo1.jpg", "..")

    with pytest.raises(ValueError):
        populated_fs.rename_file("photo1.jpg", "folder/with/slash")


def test_rename_dir_but_called_on_file_raises(populated_fs):
    with pytest.raises(FileNotFoundError):
        populated_fs.rename_dir("photo1.jpg", "photo_new.jpg")


def test_rename_file_but_called_on_dir_raises(populated_fs):
    with pytest.raises(FileNotFoundError):
        populated_fs.rename_file("photos", "images_new")


def test_delete_file_success(populated_fs):
    populated_fs.delete("text.txt")
    assert "text.txt" not in populated_fs.get_all_files()


def test_delete_dir_success(populated_fs):
    populated_fs.delete("empty")
    assert "empty" not in populated_fs.get_all_dirs()


def test_delete_nonexistent_raises(populated_fs):
    with pytest.raises(FileNotFoundError):
        populated_fs.delete("nonexistent.xyz")


def test_all_file_is_image_only_images(populated_fs):
    populated_fs.in_dir("photos")
    assert not populated_fs.all_file_is_image(recursive=False)


def test_all_file_is_image_recursive_true(populated_fs):
    # в корне есть pdf и txt -> False
    assert not populated_fs.all_file_is_image(recursive=True)


def test_all_file_is_image_empty_dir(tmp_path):
    fsm = FileSystemManager(root=tmp_path)
    assert fsm.all_file_is_image()
    assert fsm.all_file_is_image(recursive=True)


def test_all_file_is_image_only_images_recursive(populated_fs: FileSystemManager):
    print(populated_fs.get_all_dirs())
    populated_fs.in_dir("only_photos")
    assert populated_fs.all_file_is_image(recursive=True)