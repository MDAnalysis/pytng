Error Handling
==============

PyTNG is designed so that Python level exceptions can (and should) be caught by
the calling code. However, as the trajectory reading itself is done 
by Cython at the C level with the GIL released, low level trajectory reading
errors are caught by more general exceptions at the Python level.

The success of the last call to :meth:`~TNGCurrentIntegratorStep.get_blockid`, :meth:`~TNGCurrentIntegratorStep.get_positions`, :meth:`~TNGCurrentIntegratorStep.get_box` and
related methods can be checked using the :attr:`TNGCurrentIntegratorStep.read_success`
property, which returns ``True`` if the attempt to read was successful or ``False`` if not.
Success (``True``) indicates that there was data of the requested block type present at the current step,
while failure (``False``) indicates either no data present, file corruption or a file reading exception not caught by PyTNG.
These types of data reading failures are treated on an equal footing as they are in the TNG library.