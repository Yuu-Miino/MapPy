class BasicResult:
    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str({key: val for key, val in self.__dict__.items() if not key.startswith("__")})

