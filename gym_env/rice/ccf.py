import logging

import numpy as np

from rice_helpers import get_production, get_mitigation_cost, get_damages, get_abatement_cost, get_gross_output, \
    get_carbon_intensity

# JH: profiling:
do_profile = False
if do_profile:
    from line_profiler import LineProfiler

    profile = LineProfiler()
    print("!! USING LINE PROFILER, SO THIS WILL BE SLOW !!")
else:
    profile = lambda x: x  # no profiling


class CCF:

    def __init__(self, rice, env_ccf = {}):
        self.last_error_consumption_rate = None
        self.rice = rice

        if self.rice.negotiation_on:
            self.last_action_masks = {}

            # JH: renamed these and moved them to env file rice_rllib.yaml:
            self.ccf_condition_variables = env_ccf["ccf_condition_variables"]
            self.ccf_liability_variables = env_ccf["ccf_liability_variables"]
            self.num_ccf_thresholds = env_ccf["num_ccf_thresholds"]
            self.use_prioritized_ccfs = env_ccf["use_prioritized_ccfs"]
            self.allow_adjustment = env_ccf["allow_adjustment"]
            self.use_borda_scoring = env_ccf["use_borda_scoring"]
            self.verbosity_level = env_ccf["verbosity_level"]
            self.initial_ccf_parameter_value = env_ccf["initial_ccf_parameter_value"]

            self.num_ccf_condition_variables = len(self.ccf_condition_variables)
            self.num_ccf_liability_variables = len(self.ccf_liability_variables)
            self.num_ccf_condition_parameters = self.num_ccf_thresholds * self.num_ccf_condition_variables
            self.num_ccf_liability_parameters = self.num_ccf_thresholds * self.num_ccf_liability_variables
            self.num_ccf_parameters = self.num_ccf_condition_parameters + self.num_ccf_liability_parameters

            self.ccf_parameters_nvec = [self.rice.num_discrete_action_levels] * self.num_ccf_parameters

            if (self.use_prioritized_ccfs and self.allow_adjustment and self.num_ccf_thresholds >= 2):
                self.ccf_parameters_nvec += [2]  # one additional boolean control variable


    def get_actions_nvec(self):
        return self.ccf_parameters_nvec

    def reset(self, negotiation_on=False):
        if negotiation_on:
            ccf_parameters = np.zeros((self.rice.num_regions,
                                    self.num_ccf_condition_variables + self.num_ccf_liability_variables,
                                    self.num_ccf_thresholds)) + int(
                (self.rice.num_discrete_action_levels - 1) * self.initial_ccf_parameter_value
            )
            self.rice.set_global_state(
                key="ccf_parameters",
                value=ccf_parameters,
            )
            liability_profile = self.ccf_profile2liability_profile(ccf_parameters, verbose=False)
            self.rice.set_global_state(
                key="liability_profile",
                value=liability_profile,
            )
        else:
            self.rice.set_global_state(
                key="ccf_parameters",
                value=np.zeros((self.rice.num_regions, 0, 0)),
            )
            self.rice.set_global_state(
                key="liability_profile",
                value=np.zeros((self.rice.num_regions, 0)),
            )
        self.action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
        )

    @profile
    def ccf_profile2liability_profile(self, params_profile, verbose=True):
        """
        Given all regions CCFs as specified by params_profile,
        calculate the solution of the CCF mechanism.
        Depending on env.negotiation.use_ordered_ccf, this is either 
        the largest liability profile that meets all regions' conditions,
        or the supremum of all regions' top-ranked feasible liability profile
        (see below for details).
        """
        if self.use_prioritized_ccfs:
            # Use an alternative CCF mechanism that eases learning.
            # In this variant the ordering of the condition and liability 
            # variable levels chosen by the policy DOES matter.
            # Each pair of condition and liability variable level specifies an
            # "offer". The mechanism finds a "favourite" feasibly liability profile of
            # each region and then aggregates them into another feasibly liability profile.
            # 1. Find favourite feasible offer by region:
            num_regions = self.rice.num_regions
            favourites = np.zeros((num_regions, num_regions, self.num_ccf_liability_variables))
            for region_id in range(num_regions):
                for threshold_index in range(self.num_ccf_thresholds):
                    # Restrict this region's ccf to the kth threshold specified by this region:
                    restricted_params_profile = np.copy(params_profile)
                    restricted_params_profile[region_id, :, :] = params_profile[region_id, :, threshold_index].reshape((1, -1, 1))
                    # Find the largest liability profile that is feasible with this restricted ccf:
                    liability_profile = self.find_largest_feasible(restricted_params_profile)
                    # If it fulfils the considered kth threshold, store it as this region's offer:
                    if any(liability_profile[region_id, :] > 0):
                        favourites[region_id, :, :] = liability_profile
                        if verbose and self.verbosity_level > 1:
                            print(f"r{region_id} k{threshold_index} {favourites[region_id].flatten().astype('int')} {favourites[region_id].mean()}")
                        break
            # 2. Aggregate them:
            if self.use_borda_scoring:
                # a) Compute Borda scores:
                ccf_condition_variables = self.ccf_condition_variables
                num_ccf_condition_variables = self.num_ccf_condition_variables
                thresholds = params_profile[:, :num_ccf_condition_variables, :]
                offered_action_ids = params_profile[:, num_ccf_condition_variables:, :]
                if verbose and self.verbosity_level > 0:
                    print(f"  {thresholds.flatten().astype('int')} {offered_action_ids.flatten().astype('int')}")
                num_ccf_condition_variables = self.num_ccf_condition_variables
                num_ccf_thresholds = self.num_ccf_thresholds
                borda_scores = np.zeros(num_regions)
                for region1_id in range(num_regions):
                    liability_profile = favourites[region1_id, :, :]
                    # get condition variable values for this liability profile:
                    condition_variable_values_dict = self.eval_condition_variables(params_profile, liability_profile)
                    # compute the Borda score by looping over all regions:
                    for region2_id in range(num_regions):
                        region2_liabilities = liability_profile[region2_id, :]
                        # find first threshold of region2 under which the favourite of region1 is feasible:
                        region2_offers = offered_action_ids[region2_id, :, :]
                        region2_thresholds = thresholds[region2_id, :, :]
                        for threshold_index in range(num_ccf_thresholds):
                            current_offer = region2_offers[:, threshold_index]
                            current_threshold = region2_thresholds[:, threshold_index]
                            # check whether this offer is met:
                            good = np.all(region2_liabilities <= current_offer)
                            if good:
                                # check whether the threshold is met:
                                for index, var in enumerate(ccf_condition_variables):
                                    if condition_variable_values_dict[var] < current_threshold[index]:
                                        good = False
                                        break
                                if good:
                                    borda_scores[region1_id] += threshold_index
                                    break
                        if not good:
                            borda_scores[region1_id] += num_ccf_thresholds
                # b) Use the supremum of those favourites that have the smallest Borda score:
                liability_profile = np.max(favourites[np.where(borda_scores == borda_scores.min())[0], :], axis=0)
                if verbose and self.verbosity_level > 0:
                    print(f"{borda_scores.astype('int')} -> {liability_profile.flatten().astype('int')} {liability_profile.mean()} {liability_profile.std()}")
            else:
                # Simply use the supremum of all favourites:
                liability_profile = np.max(favourites, axis=0)
                if verbose and self.verbosity_level > 0:
                    print(f"--> {liability_profile.flatten().astype('int')} {liability_profile.mean()} {liability_profile.std()}")
        else:
            # Use the original CCF mechanism as described in the paper:
            # In this variant the ordering of the condition and liability 
            # variable levels chosen by the policy DOES NOT matter.
            # 1. Convert CCF params profile into an ascending sequence of conditions
            # and an ascending sequence of corresponding offers:
            sorted_params_profile = np.sort(params_profile, axis=-1)
            # 2. Find the largest feasible liability profile:
            liability_profile = self.find_largest_feasible(sorted_params_profile)
            if verbose and self.verbosity_level > 0:
                print(f"--> {liability_profile.flatten().astype('int')} {liability_profile.mean()} {liability_profile.std()}")
        return liability_profile

    def find_largest_feasible(self, params_profile):
        # 1. Start with the largest possible profile:
        liability_profile = np.zeros(
            (self.rice.num_regions, self.num_ccf_liability_variables)) + self.rice.num_discrete_action_levels - 1
        # 2. Repeatedly reduce to what regions would do at most given this, until converged:
        for it in range(100):
            last_profile = liability_profile
            liability_profile = self.eval_ccf_profile(params_profile, liability_profile)
            if np.any(liability_profile > last_profile):
                print("OOPS! liability profile increased between rounds", last_profile, liability_profile)
            if not np.any(liability_profile != last_profile):
                # no change in last iteration, so we're done:
                break
            if it >= 99:
                print("ccf did not converge at after 100 rounds", params_profile)
        return liability_profile

    @staticmethod
    def get_obs_features():
        return [
            "ccf_parameters",
            "liability_profile"
        ]

    def step(self, stage, actions):
        self.evaluate_actions(actions)

        if stage == 1:
            self.ccf_submission_step(actions)
        elif stage == 2:
            self.ccf_adjustment_step(actions)
        elif stage == 3:
            self.ccf_mechanism_step()

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def evaluate_actions(self, actions):
        if self.rice.timestep == 0:
            return

        last_action_mask = self.last_action_masks[self.rice.timestep - 1]
        for region_id in actions:
            action = actions[region_id]
            mask = last_action_mask[region_id]
            if (self.use_prioritized_ccfs and self.allow_adjustment and self.num_ccf_thresholds >= 2):
                action = action[:-2]
                mask = mask[:-2]

            # Select the valid action that is right at the boundary of the mask
            # JH @ CP: why is this called "consumption_rate", and what happens if we have several action variables rather than a single one?
            self.last_avg_liability = self.rice.get_global_state("liability_profile").mean()
            opt_consumption_rate = np.argmax(mask.reshape((-1, self.rice.num_discrete_action_levels)) < 1) - 1
            self.last_error_consumption_rate = np.abs(action[0] - opt_consumption_rate)

    def ccf_submission_step(self, actions=None):
        """
        Update Proposal States and Observations using CCF parameters
        Update Stage to 1 - Mechanism
        """
        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        # extract CCF parameters from action:
        ccf_parameters = [
            actions[region_id][self.action_offset_index: self.action_offset_index + self.num_ccf_parameters]
            .reshape((-1, self.num_ccf_thresholds))
            for region_id in range(self.rice.num_regions)
        ]

        self.rice.set_global_state("ccf_parameters", np.array(ccf_parameters))

    def ccf_adjustment_step(self, actions=None):
        """
        Each agent might move their 1st CCF point towards the "average" of all players' 1st CCF points,
        by replacing their 1st threshold by the average of everyone's 1st threshold, 
        and rescaling their 1st offer so that if all players rescale similarly, the new threshold will just be met.
        """
        if (self.use_prioritized_ccfs and self.allow_adjustment and self.num_ccf_thresholds >= 2):
            params_profile = self.rice.get_global_state("ccf_parameters", self.rice.timestep)
            ccf_condition_variables = self.ccf_condition_variables
            num_ccf_condition_variables = self.num_ccf_condition_variables
            threshold1s = params_profile[:, :num_ccf_condition_variables, 0]
            offered_action_id1s = params_profile[:, num_ccf_condition_variables:, 0]
            # extract CCF parameters from action:
            switches = [
                actions[region_id][self.action_offset_index + self.num_ccf_parameters]
                for region_id in range(self.rice.num_regions)
            ]
            if np.sum(switches) > 0:
                # some agent switches
                average_threshold1 = threshold1s.mean(axis=0)
                condition_variable_values_dict = self.eval_condition_variables(params_profile, offered_action_id1s)
                achievable_threshold1 = np.array([condition_variable_values_dict[var] for var in ccf_condition_variables])
                if self.verbosity_level > 0:
                    print(f"adj {average_threshold1.flatten()} {achievable_threshold1.flatten()} {np.array(switches).astype('int')}")
                for region_id in range(self.rice.num_regions):
                    if switches[region_id] == 1:
                        params_profile[region_id, :num_ccf_condition_variables, 0] = np.round(average_threshold1)
                        params_profile[region_id, num_ccf_condition_variables:, 0] = np.minimum(np.round(
                            # rescale 1st offer...
                            offered_action_id1s[region_id, :]
                            # ... so that it just meets the new 1st threshold: 
                            * (average_threshold1 + 1) / (achievable_threshold1 + 1)
                            # ( the "+1" prevents division by zero and adds a small safety margin )
                        ), self.rice.num_discrete_action_levels-1)
            self.rice.set_global_state("ccf_parameters", params_profile)

    def ccf_mechanism_step(self):
        """
        Calculate all regions' liabilities from all regions' submitted CCFs
        """
        params_profile = self.rice.get_global_state("ccf_parameters", self.rice.timestep)
        liability_profile = self.ccf_profile2liability_profile(params_profile)
        self.rice.set_global_state("liability_profile", liability_profile)

        if False and self.rice.timestep == 29:
            print("! t", self.rice.timestep)
            print("! c", np.sort(params_profile.astype("int"), axis=-1).reshape((-1,)))
            print("! l", liability_profile.astype("int").reshape((-1,)))

    def get_action_mask(self):
        liability_profile = self.rice.get_global_state("liability_profile", self.rice.timestep)

        if self.rice.negotiation_on:
            if ("mitigation_rate_id" in self.ccf_liability_variables
                    or "nontariff_on_clean_id" in self.ccf_liability_variables):
                mitigation_rate_ids = liability_profile[:, self.ccf_liability_variables.index("mitigation_rate_id")]
            if "consumption_rate_id" in self.ccf_liability_variables:
                consumption_rate_ids = liability_profile[:, self.ccf_liability_variables.index("consumption_rate_id")]
            if "nontariff_on_clean_id" in self.ccf_liability_variables:
                nontariff_on_clean_ids = liability_profile[:, self.ccf_liability_variables.index("nontariff_on_clean_id")]

        mask_dict = {region_id: None for region_id in range(self.rice.num_regions)}
        for region_id in range(self.rice.num_regions):
            mask = self.rice.default_agent_action_mask.copy()

            if self.rice.negotiation_on:
                if "consumption_rate_id" in self.ccf_liability_variables:
                    consumption_rate_id = int(consumption_rate_ids[region_id])
                    savings_mask = np.flip(np.array(
                        [0 for _ in range(consumption_rate_id)]
                        + [
                            1
                            for _ in range(
                                self.rice.num_discrete_action_levels - consumption_rate_id
                            )
                        ]
                    ))  # because savings rate = 1 - consumption rate
                    mask_start = 0
                    mask_end = mask_start + sum(self.rice.savings_action_nvec)
                    mask[mask_start:mask_end] = savings_mask

                if "mitigation_rate_id" in self.ccf_liability_variables:
                    mitigation_rate_id = int(mitigation_rate_ids[region_id])
                    mitigation_mask = np.array(
                        [0 for _ in range(mitigation_rate_id)]
                        + [
                            1
                            for _ in range(
                                self.rice.num_discrete_action_levels - mitigation_rate_id
                            )
                        ]
                    )
                    mask_start = sum(self.rice.savings_action_nvec)
                    mask_end = mask_start + sum(self.rice.mitigation_rate_action_nvec)
                    mask[mask_start:mask_end] = mitigation_mask

                if "nontariff_on_clean_id" in self.ccf_liability_variables:
                    nontariff_on_clean = nontariff_on_clean_ids[region_id] / self.rice.num_discrete_action_levels
                    # "nontariff_on_clean" is the fraction of clean imports that will not be taxed.
                    # "clean imports" is the fraction of imports proportional to the producer's mitigation rate.
                    # so the minimal "nontariff", the fraction of all imports that will not be taxed,
                    # is the product of "nontariff_on_clean" and producer's mitigation rate:
                    min_nontariff_ids = nontariff_on_clean * mitigation_rate_ids
                    # consequently, the maximum (effective or average) tariff on all imports is given by
                    # one minus the minimal "nontariff":
                    max_tariff_ids = self.rice.num_discrete_action_levels - 1 - min_nontariff_ids
                    tariff_masks = np.array([
                        [1 for _ in range(int(max_tariff_id) + 1)]
                        + [
                            0
                            for _ in range(
                                self.rice.num_discrete_action_levels - int(max_tariff_id) - 1
                            )
                        ]
                        for max_tariff_id in max_tariff_ids
                    ]).flatten()
                    mask_start = sum(self.rice.savings_action_nvec) + sum(self.rice.mitigation_rate_action_nvec)
                    mask_end = mask_start + tariff_masks.size
                    mask[mask_start:mask_end] = tariff_masks

            mask_dict[region_id] = mask

        # Store action mask so that we can compare it against next rounds actions
        if self.rice.negotiation_on:
            self.last_action_masks[self.rice.timestep] = mask_dict
        return mask_dict

    @profile 
    def eval_condition_variables(self, params_profile, input_profile):
        """From input_profile, calculate the aggregate indicators 
        that regions' CCFs condition on"""

        # TODO: tariffs!

        condition_variable_values_dict = {}

        if ("min_mitigation_rate_id" in self.ccf_condition_variables
                or "global_mitigation_rate_id" in self.ccf_condition_variables):
            assumed_mitigation_rate_ids = input_profile[:, self.ccf_liability_variables.index("mitigation_rate_id")]

        if "global_consumption_rate_id" in self.ccf_condition_variables:
            assumed_consumption_rate_ids = input_profile[:, self.ccf_liability_variables.index("consumption_rate_id")]

        if "min_mitigation_rate_id" in self.ccf_condition_variables:
            condition_variable_values_dict["min_mitigation_rate_id"] = assumed_mitigation_rate_ids.min()

        if ("global_mitigation_rate_id" in self.ccf_condition_variables
                or "global_consumption_rate_id" in self.ccf_condition_variables):

            productions = []
            intensities = []

            # code copied from climate_and_economy_simulation_step:
            const = self.rice.all_constants[0]
            for region_id in range(self.rice.num_regions):
                production_factor = self.rice.get_global_state(
                    "production_factor_all_regions",
                    timestep=self.rice.timestep - 1,
                    region_id=region_id,
                )
                capital = self.rice.get_global_state(
                    "capital_all_regions", timestep=self.rice.timestep - 1, region_id=region_id
                )
                labor = self.rice.get_global_state(
                    "labor_all_regions", timestep=self.rice.timestep - 1, region_id=region_id
                )
                productions.append(get_production(
                    production_factor,
                    capital,
                    labor,
                    const["xgamma"],
                ))
                intensities.append(self.rice.get_global_state(
                    "intensity_all_regions",
                    timestep=self.rice.timestep - 1,
                    region_id=region_id,
                ))

            if "global_consumption_rate_id" in self.ccf_condition_variables:
                # this is weighted by production!

                gross_outputs = []

                # code copied from climate_and_economy_simulation_step:
                prev_global_temperature = self.rice.get_global_state(
                    "global_temperature", self.rice.timestep - 1
                )
                t_at = prev_global_temperature[0]
                for region_id in range(self.rice.num_regions):
                    mitigation_rate = assumed_mitigation_rate_ids[region_id] / self.rice.num_discrete_action_levels
                    mitigation_cost = get_mitigation_cost(
                        const["xp_b"],
                        const["xtheta_2"],
                        const["xdelta_pb"],
                        self.rice.activity_timestep,
                        intensities[region_id],
                    )
                    damages = get_damages(t_at, const["xa_1"], const["xa_2"], const["xa_3"])
                    abatement_cost = get_abatement_cost(
                        mitigation_rate, mitigation_cost, const["xtheta_2"]
                    )
                    gross_outputs.append(get_gross_output(damages, abatement_cost, productions[region_id]))

#                averaging_weights = np.array(gross_outputs)  # JH: these weights depended on mitigation, so they introduced a strange interdependency between mitigation and consumption that made the CCF nonmonotonic in some cases.
                averaging_weights = np.array(productions)  # JH: these weights are independent of mitigation and should work better.
                total_weight = averaging_weights.sum()
                condition_variable_values_dict["global_consumption_rate_id"] = (np.average(
                    assumed_consumption_rate_ids,
                    weights=averaging_weights / total_weight
                ) if total_weight > 0 else assumed_consumption_rate_ids.mean())

        if "global_mitigation_rate_id" in self.ccf_condition_variables:
            # this is weighted by emissions

            averaging_weights = []
            for region_id in range(self.rice.num_regions):
                updated_intensity = get_carbon_intensity(
                    intensities[region_id],
                    const["xg_sigma"],
                    const["xdelta_sigma"],
                    const["xDelta"],
                    self.rice.activity_timestep,
                )
                averaging_weights.append(productions[region_id] * updated_intensity)
            averaging_weights = np.array(averaging_weights)
            total_weight = averaging_weights.sum()
            condition_variable_values_dict["global_mitigation_rate_id"] = (np.average(
                assumed_mitigation_rate_ids,
                weights=averaging_weights / total_weight
            ) if total_weight > 0 else assumed_mitigation_rate_ids.mean())

        return condition_variable_values_dict

    @profile
    def eval_ccf_profile(self, params_profile, input_profile):
        """
        For each region, use her CCF as specified by params_profile
        to calculate her largest offered action levels
        given all regions' hypothetical action levels as specified by input_profile.
        Assemble all these to an output_profile and return it.
        Currently, we use corner k-step CCFs, i.e., a condition is fulfilled if 
        ALL condition variables' values are no lower than the threshold;
        if the condition is fulfilled, all liability variables get their offer
        values, otherwise none of them. This enables linking different topics,
        e.g., committing to mitigate if others consume.
        """
        condition_variable_values_dict = self.eval_condition_variables(params_profile, input_profile)

        # From params_profile, check which regions' conditions are met
        # and set their output action ids accordingly:

        thresholds = params_profile[:, :self.num_ccf_condition_variables, :]
        offered_action_ids = params_profile[:, self.num_ccf_condition_variables:, :]

        num_ccf_thresholds = self.num_ccf_thresholds
        ccf_condition_variables = self.ccf_condition_variables
        output_profile = 0 * input_profile
        for region_id in range(self.rice.num_regions):
            region_thresholds = thresholds[region_id, :, :]
            # loop through all specified thresholds:
            for threshold_index in range(num_ccf_thresholds):
                current_threshold = region_thresholds[:, threshold_index]
                # check whether this threshold is met:
                good = True
                for index, var in enumerate(ccf_condition_variables):
                    if condition_variable_values_dict[var] < current_threshold[index]:
                        good = False
                        break
                if good:
                    # increase output liability to what it promised by this threshold:
                    output_profile[region_id, :] = np.maximum(output_profile[region_id, :], offered_action_ids[region_id, :, threshold_index])

        return output_profile