"""
neighbor_discovery.py
======================================================================
Modelisation des entites (P1) + Modelisation des voisins par processus
de Poisson (P2) + fonction d'utilite pour le choix du prochain saut.

Correspond aux sections 1, 2 et "Modele mathematique simplifie" du
plan de simulation :

  1. Modelisation des entites
     - D_agent  : etat (x,y,z,E,Q), actions {Selectionner Prochain Saut,
       Ajuster Trajectoire, Attendre}
     - Environnement 3D (par defaut 1000m x 1000m x 1000m)
     - Sink fixe (X_dest, Y_dest, Z_dest)

  2. Modelisation des voisins (loi de Poisson)
     - Processus de Poisson (parametre lambda) pour l'ARRIVEE de nouveaux
       voisins dans la zone de communication.
     - Loi exponentielle (parametre mu) pour le TEMPS DE SEJOUR d'un
       voisin deja a portee -- donc pour son depart.
     - Grace a la propriete d'absence de memoire de la loi exponentielle,
       la probabilite qu'un voisin disparaisse dans le prochain intervalle
       dt est CONSTANTE (independante du temps deja passe a portee) :
           P(depart dans [t, t+dt]) = 1 - exp(-mu * dt)
       C'est ce que le module expose via predict_departure_probability().

  Fonction d'utilite (choix du prochain saut) :
      U_i = w1*Proximite(i, Dest) + w2*Stabilite(i) + w3*Energie(i)

  Les poids w1, w2, w3 sont ajustables dynamiquement -- pense pour etre
  pilotes par le raisonnement CoT du LLM Core (ex: batterie faible ->
  augmenter w2 pour privilegier la stabilite plutot que la proximite).

Ce module est autonome (aucune dependance a fanet_6uav_network_sim.py)
mais concu pour s'y brancher : NeighborDiscovery peut remplacer/completer
Network.neighbors_of(), et RoutingUtility.select_best_neighbor() peut
remplacer/completer Network.best_next_hop().
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============================================================
# 1. Modelisation des entites (section 1 du plan)
# ============================================================

class DroneAction(Enum):
    SELECT_NEXT_HOP = "Selectionner_Prochain_Saut"
    ADJUST_TRAJECTORY = "Ajuster_Trajectoire"
    WAIT = "Attendre"


@dataclass
class Environment3D:
    """Bounded simulation space matching the main 500 m x 500 m FANET scenario."""
    x_max: float = 500.0
    y_max: float = 500.0
    z_max: float = 200.0

    def clamp(self, position):
        x, y, z = position
        return (
            min(max(x, 0.0), self.x_max),
            min(max(y, 0.0), self.y_max),
            min(max(z, 0.0), self.z_max),
        )

    def random_position(self, rng: random.Random):
        return (
            rng.uniform(0.0, self.x_max),
            rng.uniform(0.0, self.y_max),
            rng.uniform(0.0, self.z_max),
        )


@dataclass
class Sink:
    """Recepteur fixe (station sol / destination unique du reseau)."""
    position: tuple  # (X_dest, Y_dest, Z_dest)


@dataclass
class DroneAgentState:
    """Etat observable d'un agent-drone : D_agent = (position, Energie, Buffer)."""
    drone_id: str
    position: tuple           # (x, y, z)
    energy: float = 100.0      # E, pourcentage 0-100
    buffer_size: int = 0        # Q, nombre de paquets en attente (taille du buffer)
    last_action: DroneAction = DroneAction.WAIT

    def distance_to(self, point: tuple) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, point)))


# ============================================================
# 2. Modelisation des voisins par processus de Poisson (section 2)
# ============================================================

@dataclass
class Neighbor:
    """Un voisin actuellement a portee radio d'un agent."""
    neighbor_id: str
    position: tuple
    entered_at_tick: int        # instant d'entree dans la zone de communication
    energy: float = 100.0        # energie du voisin (necessaire pour le terme w3*Energie(i))
    link_quality: float = 1.0    # 0..1, qualite de lien (ex: derivee du RSSI)

    def sojourn_so_far(self, current_tick: int) -> int:
        return current_tick - self.entered_at_tick


class NeighborDiscovery:
    """
    Recherche/maintien de l'ensemble de voisins d'un agent, appelee a
    CHAQUE iteration (comme demande : "une methode qui cherche les
    voisins a chaque iteration").

    Deux modes d'utilisation :
      (a) mode STOCHASTIQUE PUR (par defaut) : les arrivees/departs de
          voisins suivent directement les lois de Poisson/exponentielle
          ci-dessous, sans geometrie explicite -- utile pour valider
          rapidement le module de routage independamment du moteur de
          mobilite complet.
      (b) mode GEOMETRIQUE : si vous lui passez une liste de positions
          reelles (ex: issues de GaussMarkovMobility comme dans
          fanet_6uav_network_sim.py), il derive les voisins par distance
          ET applique EN PLUS un modele de disparition/apparition
          exponentiel pour simuler des coupures de liaison transitoires
          (fading radio) meme quand deux drones restent geometriquement
          a portee -- plus realiste qu'une simple coupure au comm_range.
    """

    def __init__(self, agent_id: str, lambda_rate: float, mu_rate: float,
                 comm_range: float = 180.0, seed: int = None):
        """
        lambda_rate : taux moyen D'ARRIVEE de nouveaux voisins (voisins / unite de temps)
                       -> parametre du processus de Poisson.
        mu_rate     : taux de DEPART d'un voisin deja present
                       -> parametre de la loi exponentielle du temps de sejour.
                       Duree moyenne de sejour = 1 / mu_rate.
        comm_range  : portee radio (m), utilisee seulement en mode geometrique.
        """
        self.agent_id = agent_id
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.comm_range = comm_range
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.active_neighbors = {}   # neighbor_id -> Neighbor
        self._next_id = 0
        self.tick = 0

    # ---------- mode stochastique pur ----------

    def step_stochastic(self, dt: float = 1.0, env: Environment3D = None):
        """A appeler a chaque iteration. Fait evoluer l'ensemble de
        voisins selon Poisson (arrivees) + exponentielle (departs)."""
        self.tick += 1
        env = env or Environment3D()

        # 1. Departs : loi exponentielle, propriete d'absence de memoire
        #    -> probabilite de depart independante de sojourn_so_far()
        p_depart = self.predict_departure_probability(dt)
        for nid in list(self.active_neighbors.keys()):
            if self.rng.random() < p_depart:
                del self.active_neighbors[nid]

        # 2. Arrivees : processus de Poisson, nombre d'arrivees ~ Poisson(lambda*dt)
        n_arrivals = int(self.np_rng.poisson(self.lambda_rate * dt))
        for _ in range(n_arrivals):
            nid = f"N{self._next_id}"
            self._next_id += 1
            self.active_neighbors[nid] = Neighbor(
                neighbor_id=nid,
                position=env.random_position(self.rng),
                entered_at_tick=self.tick,
                energy=self.rng.uniform(30.0, 100.0),
                link_quality=self.rng.uniform(0.5, 1.0),
            )

        return list(self.active_neighbors.values())

    # ---------- mode geometrique (branchable sur le reste du projet) ----------

    def step_geometric(self, my_position: tuple, all_positions: dict, dt: float = 1.0):
        """all_positions : dict {drone_id: (x,y,z)} de TOUS les autres agents,
        ex. Network.positions dans fanet_6uav_network_sim.py."""
        self.tick += 1
        p_depart = self.predict_departure_probability(dt)

        # candidats geometriquement a portee
        in_range_ids = {
            did for did, pos in all_positions.items()
            if did != self.agent_id and self._distance(my_position, pos) <= self.comm_range
        }

        # departs stochastiques (fading) parmi les voisins deja actifs, meme si encore a portee
        for nid in list(self.active_neighbors.keys()):
            if nid not in in_range_ids or self.rng.random() < p_depart:
                del self.active_neighbors[nid]

        # nouvelles entrees parmi les candidats geometriques pas encore actifs
        for did in in_range_ids:
            if did not in self.active_neighbors:
                self.active_neighbors[did] = Neighbor(
                    neighbor_id=did,
                    position=all_positions[did],
                    entered_at_tick=self.tick,
                )
            else:
                self.active_neighbors[did].position = all_positions[did]

        return list(self.active_neighbors.values())

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    # ---------- prediction de disparition (propriete sans memoire) ----------

    def predict_departure_probability(self, dt: float = 1.0) -> float:
        """P(un voisin actif parte dans les prochains dt unites de temps).
        Grace a la propriete d'absence de memoire de la loi exponentielle,
        cette probabilite NE DEPEND PAS du temps deja passe a portee --
        c'est la meme pour un voisin arrive il y a 1 tick ou 50 ticks."""
        return 1.0 - math.exp(-self.mu_rate * dt)

    def expected_remaining_sojourn(self) -> float:
        """Esperance du temps de sejour RESTANT pour n'importe quel voisin actif
        (constante = 1/mu, consequence directe de l'absence de memoire)."""
        return 1.0 / self.mu_rate if self.mu_rate > 0 else float("inf")


# ============================================================
# 2bis. Outils de l'agent (P2: Get_Battery, Get_Distance, Get_Link_Quality)
# ============================================================

def get_battery(agent) -> float:
    """P2 - Outil agent : niveau de batterie courant (%). Fonctionne sur tout objet
    ayant un attribut .energy (DroneAgentState ici, ou UAVAgent dans fanet_6uav_network_sim.py)."""
    return getattr(agent, "energy", 100.0)


def get_distance(pos_a: tuple, pos_b: tuple) -> float:
    """P2 - Outil agent : distance euclidienne entre deux positions (x,y,z)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))


def get_link_quality(distance_m: float, tx_power: float = 20.0,
                       path_loss_exponent: float = 2.5, ref_loss_at_1m: float = 40.0) -> float:
    """P2 - Outil agent : qualite de lien normalisee [0,1] a partir de la distance,
    via un modele de path-loss log-distance (memes parametres que compute_rssi()
    dans fanet_6uav_network_sim.py, pour rester coherent avec le reste du projet)."""
    d = max(distance_m, 1.0)
    path_loss = ref_loss_at_1m + 10 * path_loss_exponent * math.log10(d)
    rssi_dbm = tx_power - path_loss
    # normalise une plage typique [-100dBm, -30dBm] vers [0,1]
    return max(0.0, min(1.0, (rssi_dbm + 100.0) / 70.0))


# ============================================================
# 2ter. SharedLinkGraph -- connectivite SYMETRIQUE partagee (P1/P2), utilisee par
#       fanet_6uav_network_sim.py pour alimenter A LA FOIS le systeme LLM+RL et
#       GPSR (voir gabriel_planarize ci-dessous pour la planarisation P5).
#       Contrairement a NeighborDiscovery ci-dessus (par agent, potentiellement
#       asymetrique), UN SEUL processus de naissance-mort de Poisson/exponentielle
#       par ARETE NON ORIENTEE garantit que si u voit v, v voit u -- necessaire
#       pour que la planarisation de graphe (Gabriel Graph) ait un sens.
# ============================================================

class SharedLinkGraph:
    """Shared symmetric radio graph with a hard geometric range gate.

    A pair can be active only when its 3-D distance is <= ``comm_range``.
    Inside that physical range, Poisson/exponential activation/deactivation models
    short-lived radio availability/fading.  Leaving the range immediately removes
    the link.
    """

    def __init__(self, node_ids: list, lambda_rate: float = 0.35, mu_rate: float = 0.15,
                 comm_range: float = 180.0, seed: int = None):
        self.node_ids = list(node_ids)
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.comm_range = float(comm_range)
        self.rng = random.Random(seed)
        self.edge_state = {}
        for i, u in enumerate(self.node_ids):
            for v in self.node_ids[i + 1:]:
                # None = not yet initialized while physically in range.  This avoids
                # forcing the whole FANET to start disconnected at t=0.
                self.edge_state[(u, v)] = None
        self.tick = 0

    def p_activate(self, dt: float = 1.0) -> float:
        return 1.0 - math.exp(-self.lambda_rate * dt)

    def p_deactivate(self, dt: float = 1.0) -> float:
        return 1.0 - math.exp(-self.mu_rate * dt)

    def step(self, positions: dict, dt: float = 1.0) -> list:
        """Advance connectivity once. ``positions`` must include every node, including GS.

        For a pair that is physically in range for the first time (or after it
        re-enters range), initialize the radio state from the stationary
        distribution of the two-state birth/death process:

            P(active) = lambda / (lambda + mu)

        With lambda=1.4/s and mu=0.6/s this is 0.70, instead of forcing every
        in-range edge to start OFF and waiting for a later activation event.
        """
        self.tick += 1
        pa, pd = self.p_activate(dt), self.p_deactivate(dt)
        stationary_active = (
            self.lambda_rate / (self.lambda_rate + self.mu_rate)
            if (self.lambda_rate + self.mu_rate) > 0
            else 0.0
        )

        for edge in list(self.edge_state.keys()):
            u, v = edge
            if u not in positions or v not in positions:
                self.edge_state[edge] = None
                continue

            in_range = get_distance(positions[u], positions[v]) <= self.comm_range
            if not in_range:
                # Physical range is a hard gate.  Reset to unknown so a future
                # re-entry starts from the stationary in-range distribution.
                self.edge_state[edge] = None
                continue

            state = self.edge_state[edge]
            if state is None:
                self.edge_state[edge] = self.rng.random() < stationary_active
            elif state:
                if self.rng.random() < pd:
                    self.edge_state[edge] = False
            else:
                if self.rng.random() < pa:
                    self.edge_state[edge] = True

        return [e for e, on in self.edge_state.items() if on is True]

    def neighbors_by_node(self, active_edges: list) -> dict:
        adj = {nid: [] for nid in self.node_ids}
        for (u, v) in active_edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj


def gabriel_planarize(active_edges: list, positions: dict) -> list:
    """P5 - Planarisation par Gabriel Graph : on ne garde l'arete (u,v) que si
    aucun AUTRE noeud ne se trouve strictement a l'interieur du cercle de diametre
    [u,v]. Critere de planarite standard exploite par GPSR pour garantir que le
    routage de perimetre (regle de la main droite) ne boucle pas indefiniment."""
    node_ids = list(positions.keys())
    kept = []
    for (u, v) in active_edges:
        pu, pv = positions[u], positions[v]
        mid = tuple((a + b) / 2.0 for a, b in zip(pu, pv))
        r = get_distance(pu, pv) / 2.0
        violated = False
        if r > 1e-9:
            for w in node_ids:
                if w in (u, v):
                    continue
                if get_distance(positions[w], mid) < r - 1e-6:
                    violated = True
                    break
        if not violated:
            kept.append((u, v))
    return kept




class RoutingUtility:
    """
    U_i = w1*Proximite(i, Dest) + w2*Stabilite(i) + w3*Energie(i)

    C'est le "module de routage" : entree = l'ensemble de voisins courant
    (fourni par NeighborDiscovery), sortie = le voisin choisi comme
    prochain saut. Les poids w1/w2/w3 sont le point d'ancrage pour le
    raisonnement LLM CoT (update_weights) et/ou pour un agent RL qui
    apprendrait ces poids au lieu de les fixer a la main.
    """

    def __init__(self, w1: float = 0.4, w2: float = 0.3, w3: float = 0.3):
        self.w1, self.w2, self.w3 = w1, w2, w3

    def update_weights(self, w1: float, w2: float, w3: float):
        """Point d'entree pour le LLM Core (CoT) : ex. si sa propre
        batterie est faible, il peut appeler update_weights(w1=0.2, w2=0.6, w3=0.2)
        pour privilegier la stabilite du lien plutot que la proximite pure."""
        total = w1 + w2 + w3
        if total > 0:  # normalisation pour rester une combinaison convexe
            self.w1, self.w2, self.w3 = w1 / total, w2 / total, w3 / total

    @staticmethod
    def proximity_score(neighbor_pos: tuple, dest_pos: tuple, max_distance: float) -> float:
        """1.0 si le voisin est au niveau du sink, decroit lineairement avec la distance."""
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(neighbor_pos, dest_pos)))
        return max(0.0, 1.0 - min(d / max_distance, 1.0))

    @staticmethod
    def stability_score(neighbor: Neighbor, discovery: NeighborDiscovery) -> float:
        """Stabilite = probabilite que le voisin RESTE a portee au prochain pas
        (1 - probabilite de depart). Constante par la propriete sans memoire,
        donc identique pour tous les voisins actifs a un instant donne --
        c'est une simplification volontaire assumee par le modele."""
        return 1.0 - discovery.predict_departure_probability(dt=1.0)

    @staticmethod
    def energy_score(neighbor: Neighbor) -> float:
        return max(0.0, min(neighbor.energy / 100.0, 1.0))

    def compute_utility(self, neighbor: Neighbor, dest_pos: tuple, max_distance: float,
                          discovery: NeighborDiscovery) -> float:
        prox = self.proximity_score(neighbor.position, dest_pos, max_distance)
        stab = self.stability_score(neighbor, discovery)
        energy = self.energy_score(neighbor)
        return self.w1 * prox + self.w2 * stab + self.w3 * energy

    def select_best_neighbor(self, neighbors: list, dest_pos: tuple, max_distance: float,
                               discovery: NeighborDiscovery):
        """Entree : l'ensemble de voisins courant (liste de Neighbor).
        Sortie : (meilleur_voisin, U_i) selon la fonction d'utilite, ou (None, 0.0)
        si aucun voisin n'est disponible -- c'est ce resultat qui alimente
        l'action DroneAction.SELECT_NEXT_HOP du module de routage RL."""
        if not neighbors:
            return None, 0.0
        scored = [(n, self.compute_utility(n, dest_pos, max_distance, discovery)) for n in neighbors]
        return max(scored, key=lambda pair: pair[1])


# ============================================================
# 4. Demonstration autonome
# ============================================================

def _demo(steps: int = 10):
    env = Environment3D(x_max=1000.0, y_max=1000.0, z_max=1000.0)
    sink = Sink(position=(1000.0, 1000.0, 0.0))
    max_distance = math.sqrt(env.x_max ** 2 + env.y_max ** 2 + env.z_max ** 2)

    agent = DroneAgentState(drone_id="D1", position=(500.0, 500.0, 200.0), energy=80.0, buffer_size=3)

    # lambda=0.8 nouveau voisin/tick en moyenne, mu=0.2 -> duree moyenne de sejour = 5 ticks
    discovery = NeighborDiscovery(agent_id=agent.drone_id, lambda_rate=0.8, mu_rate=0.2, seed=42)
    utility = RoutingUtility(w1=0.4, w2=0.3, w3=0.3)

    print(f"Duree moyenne de sejour d'un voisin (1/mu) = {discovery.expected_remaining_sojourn():.1f} ticks")
    print(f"P(depart au prochain tick), constante (sans memoire) = {discovery.predict_departure_probability():.3f}\n")

    for t in range(1, steps + 1):
        neighbors = discovery.step_stochastic(dt=1.0, env=env)

        # exemple d'ajustement dynamique des poids par le "raisonnement CoT" :
        # si l'energie de l'agent est faible, on privilegie la stabilite du lien
        if agent.energy < 40:
            utility.update_weights(w1=0.2, w2=0.6, w3=0.2)
        else:
            utility.update_weights(w1=0.4, w2=0.3, w3=0.3)

        best_neighbor, best_u = utility.select_best_neighbor(neighbors, sink.position, max_distance, discovery)
        agent.last_action = DroneAction.SELECT_NEXT_HOP if best_neighbor else DroneAction.WAIT

        print(f"tick {t}: {len(neighbors)} voisin(s) actif(s) | "
              f"action={agent.last_action.value} | "
              f"meilleur_voisin={best_neighbor.neighbor_id if best_neighbor else None} "
              f"(U={best_u:.3f})")

        agent.energy = max(0.0, agent.energy - 1.5)  # decharge simulee


if __name__ == "__main__":
    _demo(steps=10)