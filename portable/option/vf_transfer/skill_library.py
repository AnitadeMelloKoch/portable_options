import logging
import os
import numpy as np
import gin
import random
import torch
import pickle

from portable.option.vf_transfer.vf_transfer_option import VFTransferAgent


@gin.configurable
class SkillLibrary():
    def __init__(self,
                 skill_factory,
                 similarity_min,
                 init_threshold=0.6):
        # a list of vf transfer options
        self.skills = []
        self.skill_factory = skill_factory
        self.similarity_min = similarity_min
        self.init_threshold = init_threshold

    def check_init(self, state):
        """Return list of (skill_idx, probability) for skills whose initiation
        probability on state meets init_threshold."""
        candidates = []
        for idx, skill in enumerate(self.skills):
            prob = skill.initiation(state)
            if prob >= self.init_threshold:
                candidates.append((idx, prob))
        return candidates

    def run_task(self, env):
        """Select the most initiated skill (or create a new one) and run one episode.

        Peeks at the initial state to score each skill's initiation probability,
        commits to the best candidate, then runs the full episode. env.reset() is
        called again inside run_episode, which is fine for fixed-seed environments.
        """
        init_state, _ = env.reset()
        candidates = self.check_init(init_state)

        if candidates:
            best_idx = max(candidates, key=lambda x: x[1])[0]
            skill = self.skills[best_idx]
            logging.info(f"SkillLibrary: using skill {best_idx} "
                         f"(prob={candidates[best_idx][1]:.3f})")
        else:
            skill = self.skill_factory()
            self.skills.append(skill)
            logging.info(f"SkillLibrary: created new skill (total={len(self.skills)})")

        return skill.run_episode(env)

    def consolidate_v(self, states, values):
        """Assign task experience to the most similar skill's meta VF, or create a new skill.

        Similarity between the task V distribution and each skill's meta VF is computed
        as exp(-KL(V_task || V_meta)), so it lives in (0, 1]. If the best match exceeds
        similarity_min the task is assigned to that skill; otherwise a new skill is created.

        Args:
            states: list of observations from data collection
            values: tuple (means, stds) — V_task per state from the DQN ensemble
        """
        if not self.skills:
            skill = self.skill_factory()
            self.skills.append(skill)
            logging.info("SkillLibrary.consolidate_v: no skills yet, created first skill")
            return len(self.skills) - 1

        means, stds = values
        device = self.skills[0].meta_vf.device

        states_t = torch.stack([
            torch.tensor(s, dtype=torch.float32) for s in states
        ]).to(device)
        means_t = torch.tensor(means, dtype=torch.float32).to(device)
        stds_t = torch.tensor(stds, dtype=torch.float32).to(device)

        best_idx, best_sim = -1, -1.0
        for idx, skill in enumerate(self.skills):
            with torch.no_grad():
                v_meta_mean, v_meta_std = skill.meta_vf.query(states_t)
                v_meta_mean = v_meta_mean.squeeze(1)
                v_meta_std = v_meta_std.squeeze(1)
            kl = self._gaussian_kl(means_t, stds_t, v_meta_mean, v_meta_std)
            sim = torch.exp(-kl.mean()).item()
            if sim > best_sim:
                best_sim, best_idx = sim, idx

        if best_sim >= self.similarity_min:
            logging.info(f"SkillLibrary.consolidate_v: assigned to skill {best_idx} "
                         f"(similarity={best_sim:.3f})")
            return best_idx
        else:
            skill = self.skill_factory()
            self.skills.append(skill)
            logging.info(f"SkillLibrary.consolidate_v: no match (best_sim={best_sim:.3f}), "
                         f"created new skill (total={len(self.skills)})")
            return len(self.skills) - 1

    def _gaussian_kl(self, mu_p, sigma_p, mu_q, sigma_q):
        """KL(p || q) per state, p = V_task, q = V_meta."""
        sigma_p = sigma_p + 1e-8
        sigma_q = sigma_q + 1e-8
        return (torch.log(sigma_q / sigma_p) +
                (sigma_p.pow(2) + (mu_p - mu_q).pow(2)) / (2 * sigma_q.pow(2)) - 0.5)

    def __len__(self):
        return len(self.skills)

    