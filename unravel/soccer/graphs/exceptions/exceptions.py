import warnings
from packaging import version
from unravel import __version__ as installed_version

class VersionMismatchError(Exception):
    """Exception raised for major or minor version mismatches."""
    pass

class VersionChecker:
    """Class to check and warn about version mismatches."""

    @classmethod
    def check_versioning(cls, config_version):
        """
        Check if the installed version matches the configuration version.
        
        Args:
            config_version (str): Version string from the configuration
            
        Raises:
            VersionMismatchError: If the major or minor version differs.
        Warns:
            VersionMismatchWarning: If only the patch version differs.
        """
        installed_ver = version.parse(installed_version)
        config_ver = version.parse(config_version)

        # Extract major, minor, and patch
        installed_major, installed_minor, installed_patch = installed_ver.release
        config_major, config_minor, config_patch = config_ver.release

        release_notes_url = "https://github.com/UnravelSports/unravelsports/releases"
        
        if (installed_major, installed_minor) != (config_major, config_minor):
            raise VersionMismatchError(
                f"Version mismatch detected: You are using unravelsports v{installed_version}, "
                f"but your configuration was created with v{config_version}.\n"
                f"This may cause unexpected behavior or incompatibilities.\n"
                f"Please check the release notes: {release_notes_url}"
            )
        
        if installed_patch != config_patch:
            warnings.warn(
                f"Patch version mismatch detected: Installed v{installed_version}, "
                f"but config was created with v{config_version}.\n"
                f"While this is usually safe, please check the release notes: {release_notes_url}",
                UserWarning
            )

class MissingDatasetError(Exception):
    pass


class MissingLabelsError(Exception):
    pass


class IncorrectDatasetTypeError(Exception):
    pass


class IncorrectKloppyOrientationError(Exception):
    pass


class KeyMismatchError(Exception):
    pass
