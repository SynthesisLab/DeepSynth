from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any


@dataclass(order=True, frozen=True)
class CostTuple:
    cost: float
    combinations: List[List[int]]

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combinations})"


class CDQueue:
    """
    Init:
        push*

    Usage:
        pop
        push*
        update

    """

    def __init__(self, maxi: int, k: int) -> None:
        # multiply otherwise when you get exactly maxi then it is equal to 0
        self.maxi = maxi * (k + 1) / k
        self.k = k + 1
        self.mini: Optional[float] = None
        self.cells: List[Tuple[int, Optional[Union[CostTuple, Any]]]] = [
            (0, None) for _ in range(self.k)
        ]
        self.translation = 0
        self.nelements = 0
        self.start: Optional[float] = None
        self.n = 0
        # number of times the window was moved or doubled because an element did not fit
        self.grows = 0
        self.clear()

    def update(self) -> None:
        """
        Update its internal representation, should be done after all elements have been pushed.
        """
        if self.nelements > 0:
            while self.cells[self.translation][0] == 0:
                self.translation = (self.translation + 1) % self.k
                self.n += 1
            if self.nelements == 1:
                self.mini = self.cells[self.translation][1].cost  # type: ignore
                self.start = self.mini
                self.n = 0
            else:
                self.mini = self.start + self.maxi * self.n / self.k  # type: ignore

    def clear(self) -> None:
        """
        Clear this queue of all of its elements.
        """
        self.mini = None
        self.cells = [(0, None) for _ in range(self.k)]
        self.translation = 0
        self.nelements = 0
        self.start = None
        self.n = 0

    def push(self, element: CostTuple) -> None:
        # print("PUSH")
        if self.mini is None:
            self.mini = element.cost
            self.start = self.mini
            self.n = 0
        if element.cost < self.mini:
            # should not happen if elements are pushed in the order described above,
            # the window is moved instead of placing the element in a wrong cell
            self.__rebuild__(element.cost, self.mini + self.maxi - element.cost)
        # an element at distance maxi would land in the current cell, hence >=
        elif element.cost - self.mini >= self.maxi:
            self.__rebuild__(self.mini, element.cost - self.mini)
        # a = f"[PUSH] {F.LIGHTYELLOW_EX}BEFORE{F.RESET}:{self}"
        # print(f"[PUSH] {F.LIGHTYELLOW_EX}BEFORE{F.RESET}:", self)
        # b = f"\t\t\t{element}"
        # print("\t\t\t", element)
        self.__push__(
            element, element.cost - self.mini, self.maxi, self.cells, self.translation
        )
        # if x or abs(element.cost - 11.040864126152853) <= 1e-3:
        #     print("n=", self.n, "unit:", self.maxi / self.k)
        #     print(a)
        #     print(b)
        #     print(f"[PUSH] {F.GREEN}AFTER{F.RESET}:", self)

    def __rebuild__(self, mini: float, width: float) -> None:
        """
        Move the lower bound of the window to mini, double its width until it is larger than width,
        and push back all elements.
        When called from push, the lower bound is kept or lowered, so that elements pushed after a pop still fit.
        """
        elems = sorted(self.__elements__(self.cells), key=lambda e: e.cost)
        if elems:
            width = max(width, elems[-1].cost - mini)
        maxi = self.maxi
        while maxi <= width:
            maxi *= 2
        self.clear()
        self.maxi = maxi
        self.mini = mini
        self.start = mini
        for elem in elems:
            self.__push__(
                elem, elem.cost - mini, self.maxi, self.cells, self.translation
            )
        self.grows += 1

    def __elements__(self, cells: List) -> List[CostTuple]:
        out = []
        for nelem, val in cells:
            if val is None:
                continue
            if nelem == 1:
                out.append(val)
            else:
                out += self.__elements__(val)
        return out

    def __push__(
        self,
        element: CostTuple,
        cost: float,
        maxi: float,
        cells: List,
        translation: int,
        add: bool = True,
    ) -> None:
        stack: List[Tuple[List[List[int]], int]] = []
        while True:
            unit = maxi / self.k
            # clamp: with floating point errors an element on the upper boundary of a cell
            # can be pushed into this cell with an offset equal to its width,
            # it must then go into the last sub-cell and not wrap around to the first one
            lbi = min(max(int(cost / maxi * self.k), 0), self.k - 1)
            index = (lbi + translation) % self.k
            # assert cost >= 0, f"cost:{cost} element:{element} queue:{self}"
            nelem, val = cells[index]
            # print("\tfound:", nelem, "val=", val)
            if nelem == 0:
                # print("\t\tfilling...")
                cells[index] = (1, element)
                if add:
                    self.nelements += 1
                    for c, i in stack:
                        c[i][0] += 1
                return

            elif nelem == 1:
                if abs(val.cost - element.cost) > 1:
                    # print(f"\t\tsplitting with {val}... diff={abs(val.cost - element.cost)} ")

                    cells[index] = [2, [(0, None) for _ in range(self.k)]]
                    self.__push__(
                        val,
                        cost + val.cost - element.cost - lbi * unit,
                        unit,
                        cells[index][1],
                        0,
                        add=False,
                    )
                    self.__push__(
                        element,
                        cost - lbi * unit,
                        unit,
                        cells[index][1],
                        0,
                        add=False,
                    )
                    if add:
                        self.nelements += 1
                        for c, i in stack:
                            c[i][0] += 1
                    return
                else:
                    val.combinations.extend(element.combinations)
                    # print("\t\tmerging...")
                    return
            else:
                # print("\t\trecursive...")
                stack.append((cells, index))
                cost = cost - lbi * unit
                maxi = unit
                cells = val
                translation = 0
        return

    def __cleanup__(self, cells: List[Tuple[int, Any]], index: int) -> None:
        # print(f"[CLEANUP] {F.LIGHTBLUE_EX}BEFORE{F.RESET}:", cells[index])
        n, _ = cells[index]
        if n > 1:
            if n == 2:
                cells[index] = (1, self.__pop__(cells[index]))
                # print("AFTER:", self)
            else:
                cells[index][0] -= 1  # type: ignore
        else:
            cells[index] = (0, None)
        # print("[CLEANUP] AFTER:", cells[index])

    def pop(self) -> CostTuple:
        popped = self.__pop__(self.cells[self.translation])
        self.__cleanup__(self.cells, self.translation)
        self.nelements -= 1
        self.last_pop = popped
        return popped  # type: ignore

    def __pop__(self, cells: Tuple[int, Any]) -> Optional[CostTuple]:
        nelems, val = cells
        if nelems <= 1:
            return val  # type: ignore
        for i, elem in enumerate(val):
            n, v = elem
            if n > 0:
                if n == 1:
                    val[i] = (0, None)
                    return v  # type: ignore
                else:
                    popped = self.__pop__(elem)
                    if popped is not None:
                        self.__cleanup__(val, i)
                        return popped
                    else:
                        val[i] = (0, None)
        return None

    def peek(self) -> CostTuple:
        # term, content = self.cells[self.translation]
        # assert content is not None
        popped = self.__peek__(self.cells[self.translation])
        # assert popped is not None
        return popped  # type: ignore

    def __peek__(self, cells: Tuple) -> Optional[CostTuple]:
        nelems, val = cells
        if nelems <= 1:
            return val  # type: ignore
        for elem in val:
            n, v = elem
            if v is not None:
                if n == 1:
                    return v  # type: ignore
                else:
                    popped = self.__peek__(elem)
                    if popped is not None:
                        return popped
        return None

    def size(self) -> int:
        return sum(self.__size__(el) for el in self.cells)

    def __size__(self, cell: Tuple) -> int:
        terminal, val = cell
        if val is None:
            return 0
        elif terminal == 1:
            return 1
        else:
            return sum(max(1, self.__size__(elem)) for elem in val)

    def is_empty(self) -> bool:
        return self.nelements == 0

    def __repr__(self) -> str:
        ordered = self.cells[self.translation :] + self.cells[: self.translation]
        out = f"CDQueue[size={self.nelements}/{self.size()}, mini={self.mini}, maxi={self.mini + self.maxi * (self.k / (self.k + 1))}/{self.mini + self.maxi}, k={self.k}]\n\t{ordered}"  # type: ignore
        return out

    def __len__(self) -> int:
        return self.nelements
