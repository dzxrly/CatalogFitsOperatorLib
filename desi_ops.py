from dl import authClient as ac
from dl import queryClient as qc
from sparcl.client import SparclClient


class DESIOps:
    def __init__(
        self,
        astro_datalab_email: str,
        astro_datalab_username: str,
        astro_datalab_password: str,
        verbose: bool = False,
        show_curl: bool = False,
        connect_timeout_sec=1.1,
        read_timeout_sec=90 * 60,
    ):
        """
        Initialize the DESIOps class with the necessary parameters.

        Parameters
        ----------
        astro_datalab_email : str
            The email associated with the Astro Data Lab account.
        astro_datalab_username : str
            The username for the Astro Data Lab account.
        astro_datalab_password : str
            The password for the Astro Data Lab account.
        verbose : bool, optional
            If True, enables verbose output (default is False).
        show_curl : bool, optional
            If True, shows curl commands (default is False).
        connect_timeout_sec : float, optional
            The connection timeout in seconds (default is 1.1).
        read_timeout_sec : float, optional
            The read timeout in seconds (default is 90 minutes).
        """
        super().__init__()
        self.sparclClient = SparclClient(
            verbose=verbose,
            show_curl=show_curl,
            connect_timeout=connect_timeout_sec,
            read_timeout=read_timeout_sec,
        )
        self.authClient = ac
        self.queryClient = qc
        self.verbose = verbose

        self.sparclClient.login(astro_datalab_email, astro_datalab_password)
        self.authClient.login(astro_datalab_username, astro_datalab_password)

    def get_info(self, catalog: str) -> str:
        """
        Get the info of a DESI catalog.

        Parameters
        ----------
        catalog : str
            The name of the DESI catalog to query.

        Returns
        -------
        str
            A string representation of the info in the catalog.
        """
        return self.queryClient.schema(value=catalog)
