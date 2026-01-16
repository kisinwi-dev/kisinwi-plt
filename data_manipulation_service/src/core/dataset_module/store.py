from logging_ import get_logger
from .case import DatasetImporter
from .filesystem import FileSystemManager, ArchiveManager
from .models import ClassInfo, VersionInfo, DatasetInfo

logger = get_logger(__name__)

class Store:
    def __init__(
            self,
            fsm: FileSystemManager | None = None
        ):
        self._fsm = fsm if fsm else FileSystemManager()

    # ------------------ Dataset info methods ------------------

    def get_dataset_name(self) -> list[str]:
        self._fsm.reset()
        dataset_list = self._fsm.get_all_dir()
        return dataset_list

    def get_dataset_version_name(self, dataset_name: str) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        version_list = self._fsm.get_all_dir()
        return version_list        

    def get_dataset_classes_name(
            self, 
            dataset_name: str,
            dataset_version: str,
        ) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, dataset_version])
        classes_list = self._fsm.get_all_dir()
        return classes_list
    
    def get_dataset_files_name(
            self,
            dataset_name: str,
            dataset_version: str,
            class_name: str
        ) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, dataset_version, class_name])
        return self._fsm.get_all_file()
    
    def get_dataset_info(
            self,
            dataset_name: str
        ):
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        versions = self._fsm.get_all_dir()
        info_versions = []
        for version in versions:
            self._fsm.in_dir(version)
            classes = self._fsm.get_all_dir()
            info_classes = []
            for class_ in classes:
                self._fsm.in_dir(class_)
                files = self._fsm.get_all_file()
                info_classes.append(
                    ClassInfo(
                        name=class_,
                        description=None,
                        count_files=len(files),
                        type_files=None
                    )
                )
                self._fsm.out_dir()
            
            info_versions.append(
                VersionInfo(
                    name=version,
                    description=None,
                    classes=info_classes
                )
            )
            self._fsm.out_dir()
        
        return DatasetInfo(
            name=dataset_name,
            description=None,
            versions=info_versions
        )

    # ------------------ Dataset management ------------------

    def set_new_dataset(
            self, 
            dataset_name: str,
            archive_name: str,
            dataset_type: str,
            dataset_task: str
        ):
        logger.debug(
            f'Start create new dataset(name={dataset_name}, archive_name={archive_name})'
        )
        importer = DatasetImporter(
            datasets_fsm=self._fsm,
            archive_manager=ArchiveManager()
        )
        importer.import_dataset(
            dataset_name=dataset_name, 
            archive_name=archive_name,
            dataset_type=dataset_type, 
            dataset_task=dataset_task
        )

    def drop_dataset(self, dataset_name: str):
        self._fsm.reset()
        self._fsm.drop_dir(dataset_name)

    def rename_dataset(self, old_name: str, new_name: str):
        self._fsm.reset()
        self._fsm.rename_dir(old_name, new_name)

    def rename_dataset_version(self, dataset_name: str, old_version: str, new_version: str):
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        self._fsm.rename_dir(old_version, new_version)

    def set_new_dataset_version(self, dataset_name: str, version_name: str):
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        (self._fsm.worker_path / version_name).mkdir(exist_ok=False)

    # ------------------ Class management ------------------

    def set_new_class(
            self, 
            dataset_name: str, 
            version_name: str, 
            class_name: str
        ):
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, version_name])
        (self._fsm.worker_path / class_name).mkdir(exist_ok=False)

    def drop_class(
            self, 
            dataset_name: str, 
            version_name: str, 
            class_name: str
        ):
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, version_name])
        self._fsm.drop_dir(class_name)

    def rename_class(
            self, 
            dataset_name: str, 
            version_name: str, 
            old_class: str, 
            new_class: str
        ):
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, version_name])
        self._fsm.rename_dir(old_class, new_class)

    def validation_dataset(self):
        pass
