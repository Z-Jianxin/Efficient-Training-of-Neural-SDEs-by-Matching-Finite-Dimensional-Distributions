import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils.helper_functions.plot_helper_functions import make_grid
from tqdm import tqdm
import math
from src.gan.discriminators import SigKerMMDDiscriminator, TruncatedDiscriminator, CDEDiscriminator, FDDiscriminator, RFDDiscriminator
from src.rBergomi import rBergomi
from src.gan import sde
from src.gan.base import preprocess_real_data, get_real_data, get_synthetic_data, get_scheduler, stopping_criterion
from src.utils.helper_functions.data_helper_functions import subtract_initial_point
from src.utils.transformations import Transformer


def plot_scaled_increment_values(sigker_mmd_config, infinite_train_dataloader):
    sigmas = sigker_mmd_config.get("sigma")
    kernel_type = sigker_mmd_config.get("kernel_type")

    pow_func = lambda x, y: torch.pow(x, y) / math.factorial(y)
    bnd_func = lambda x, y: pow_func(torch.max(torch.sum(torch.abs(torch.diff(x, axis=1)), axis=1)), y)

    do_theo_bnd = False

    fig_list = []

    with torch.no_grad():
        paths, = next(infinite_train_dataloader)
        _, _, dims = paths.shape
        dims -= 1
        
        paths = paths.cpu()
        
        if not isinstance(sigmas, list):
            sigmas = np.array([sigmas])
        
        for k in range(dims):
            ex_title = f", dim = {k+1}" if dims != 1 else ""

            fig, axes = plt.subplots(1, len(sigmas), figsize=(6 * len(sigmas), 3))
            fig_list.append(fig)

            if len(sigmas) == 1:
                axes = np.array([axes])

            for ax, sig in zip(axes, sigmas):
                ex_sig = f", sigma = {sig}" if kernel_type == "rbf" else ""

                increments = torch.abs(paths[:, -1, k+1] - paths[:, 0, k+1])
                scaled_increments = increments / np.sqrt(sig) if kernel_type == "rbf" else increments

                powers = np.arange(1, 10).astype(int)

                moment_terms = torch.tensor([[pow_func(inc, d) for d in powers] for inc in scaled_increments])
                moment_means = moment_terms.mean(axis=0)
                moment_stds = moment_terms.std(axis=0)
                
                ax.plot(powers, moment_means, color="dodgerblue", alpha=0.75, label="moment_means")
                if do_theo_bnd:
                    level_bds = [bnd_func(paths[..., k+1], d) for d in powers]
                    ax.plot(powers, level_bds, color="tomato", linestyle="dashed", alpha=0.75, label="theo_bnd")
                    
                ax.fill_between(powers, moment_means - moment_stds, moment_means + moment_stds, color="dodgerblue", alpha=0.25)
                make_grid(axis=ax)
                ax.set_title("Scaled increment values (moment ratio)" + ex_sig + ex_title, fontsize="small")
                ax.legend(fontsize="small")
    return fig_list

def adjust_generator_parameters(generator, generator_config, learning_type, init_mult1, init_mult2_dr, init_mult2_df, **kwargs):
    with torch.no_grad():
        if (not generator_config.get("fixed")) or (learning_type == "returns"):
            for prm in generator._initial.parameters():
                prm *= init_mult1

        for name, prm in generator._func.named_parameters():
            if "_drift" in name:
                prm.data *= init_mult2_dr
            else:
                prm.data *= init_mult2_df
    return generator

def plot_generated_vs_real(output_dim, data_type, forex_pairs, infinite_train_dataloader, generator, ts, batch_size, subtract_start, subtract_initial_point):
    dims = output_dim if data_type != "forex" else len(forex_pairs)
    
    fig, axes = plt.subplots(dims, 1, figsize=(6, dims*3))
    
    with torch.no_grad():
        y, = next(infinite_train_dataloader)
        x  = generator(ts, batch_size)
        
        if subtract_start:
            y  = subtract_initial_point(y)
            x  = subtract_initial_point(x)
        
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    for k, ax in enumerate(axes):

        N = 32
        x_plot = x[:N, :, k+1].cpu()
        y_plot = y[:N, :, k+1].cpu()

        generated_first = True
        real_first      = True

        for xi, yi in zip(x_plot, y_plot):
            g_kwargs = {"label": "generated"} if generated_first else {}
            r_kwargs = {"label": "real"} if real_first else {}

            ax.plot(xi, color="tomato", alpha=0.5, **g_kwargs)
            ax.plot(yi, color="dodgerblue", alpha=0.5, **r_kwargs)

            generated_first = False
            real_first      = False

        ax.legend()
        make_grid(axis=ax)
        ax_title = "" if dims == 1 else f"Dim {k+1}"
        ax.set_title(ax_title, fontsize="small")
    fig.suptitle("Initialisation of $G$ against real data")
    plt.tight_layout()
    return fig

def plot_mmd_histogram(batch_size, infinite_train_dataloader, subtract_start, subtract_initial_point, transformer, discriminator, mmd_ci):
    with torch.no_grad():
        mmd_atoms = batch_size
        true_mmds = np.zeros(mmd_atoms)

        for i in tqdm(range(mmd_atoms)):
            x, = next(infinite_train_dataloader)
            y, = next(infinite_train_dataloader)

            if subtract_start:
                x = subtract_initial_point(x)
                y = subtract_initial_point(y)

            x = transformer(x)
            y = transformer(y)

            true_mmds[i] = discriminator(x, y.detach())

        ci = sorted(true_mmds)[int(mmd_ci * mmd_atoms)]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(sorted(true_mmds), bins=int(mmd_atoms / 10), alpha=0.6, color="dodgerblue", density=True)
        make_grid(axis=ax)
        ax.set_title(f"{100 * mmd_ci:.0f}% CI: {ci:.5e}")
    return fig

def create_discriminator(discriminator_type, data_size, discr_config):
    discriminator_list = ["SigKerMMDDiscriminator", "TruncatedDiscriminator", "CDEDiscriminator", "FDDiscriminator", "FDADiscriminator", "RFDDiscriminator"]
    discr_config["adversarial"] = False
    discr_config["path_dim"] = data_size
    discr_config["data_size"] = data_size
    if discriminator_type in discriminator_list:
        discriminator = eval(discriminator_type)(**discr_config)
    else:
        discriminator = None
        print(f"Discriminator {discriminator} does not exist.")
        print(f"Choose from {', '.join(discriminator_list)}")
    return discriminator

def setup_optimizers_and_schedulers(generator, discriminator, config):
    # Setup generator optimizer
    generator_optimiser = getattr(torch.optim, config["gen_optim"])(
        generator.parameters(), lr=config["generator_lr"], weight_decay=config["weight_decay"]
    )
    generator_optimiser.zero_grad()

    # Setup discriminator optimizer
    #if config["adversarial"] or (config["discriminator_type"] == "wasserstein_cde"):
    if isinstance(discriminator, CDEDiscriminator):
        discriminator_optimiser_ = getattr(torch.optim, config["disc_optim"])
        discriminator_optimiser = discriminator_optimiser_(
            discriminator.parameters(),
            lr=config["discriminator_lr"],
            weight_decay=config["weight_decay"]
        )
        discriminator_optimiser.zero_grad()
    else:
        discriminator_optimiser = None

    # Setup learning rate schedulers
    if config["adapting_lr"]:
        if config["adapting_lr_type"] == "LambdaLR":
            f_lmd = lambda epoch: (1 - config["lambda_lr_const"]) * np.power(epoch + 1., config["poly_exponent_smoother"]) + config["lambda_lr_const"]
            adpt_kwargs = {"lr_lambda": f_lmd}
        elif config["adapting_lr_type"] == "StepLR":
            adpt_kwargs = {"gamma": config["gamma_lr"], "step_size": config["steps_lr"]}
        elif config["adapting_lr_type"] == "MultiplicativeLR":
            adpt_kwargs = {"lr_lambda": lambda epoch: config["mult_const"]}
        elif config["adapting_lr_type"] == "OneCycleLR":
            adpt_kwargs = {
                "max_lr": config["max_lr"],
                "total_steps": config["total_steps"],
                "pct_start": config["pct_start"],
                "anneal_strategy": config["anneal_strategy"],
                "div_factor": config["div_factor"]
            }

        g_scheduler, d_scheduler = get_scheduler(
            generator_optimiser,
            discriminator_optimiser,
            config["adapting_lr_type"],
            isinstance(discriminator, CDEDiscriminator), #False, #config["adversarial"],
            **adpt_kwargs
        )
    else:
        g_scheduler, d_scheduler = None, None

    return generator_optimiser, discriminator_optimiser, g_scheduler, d_scheduler

def get_data(data_type, data_config, device):
    # Extracting values from the dictionary
    dataset_size = data_config["dataset_size"]
    path_length = data_config["path_length"]
    batch_size = data_config["batch_size"]
    stride_length = data_config["stride_length"]
    learning_type = data_config["learning_type"]
    time_add_type = data_config["time_add_type"]
    initial_point = data_config["initial_point"]
    tt_split = data_config["tt_split"]
    forex_pairs = data_config["forex_pairs"]
    frequency = data_config["frequency"]
    filter_extremal_paths = data_config["filter_extremal_paths"]
    filter_extremal_pct = data_config["filter_extremal_pct"]
    normalisation = data_config["normalisation"]
    filter_by_time = data_config["filter_by_time"]
    scale = data_config["scale"]
    gen_sde_method = data_config["gen_sde_method"]
    gen_sde_dt_scale = data_config["gen_sde_dt_scale"]
    gen_sde_integration_type = data_config["gen_sde_integration_type"]
    sde_parameters = data_config["sde_parameters"]
    output_dim = data_config["output_dim"]
    end_time = data_config["end_time"]
    do_transforms = data_config["do_transforms"]
    transformations = data_config["transformations"]
    transformation_args = data_config["transformation_args"]
    preserve_order = data_config["preserve_order"]

    # Data kwargs
    data_kwargs = {
        "dataset_size": dataset_size,
        "path_length": path_length,
        "batch_size": batch_size,
        "step_size": stride_length,
        "learning_type": learning_type,
        "time_add_type": time_add_type,
        "initial_point": initial_point,
        "train_test_split": tt_split
    }

    # Real data kwargs
    real_data_kwargs = {
        "data_type": data_type,
        "pairs": forex_pairs,
        "frequency": frequency,
        "filter_extremal_paths": filter_extremal_paths,
        "filter_extremal_pct": filter_extremal_pct,
        "preserve_order": preserve_order
    }

    if data_type.lower() in ["forex", "energy", "indices", "bonds", "forex_metals"]:  # Real data arguments
        np_train_data, np_test_data = preprocess_real_data(data_kwargs, real_data_kwargs)
        #print(np_test_data[1, 1, :])
        ts, data_size, train_dataloader = get_real_data(
            np_train_data,
            batch_size,
            dataset_size,
            device,
            time_add_type=time_add_type,
            normalisation=normalisation,
            filter_by_time=filter_by_time,
            initial_point=False,
            scale=scale
        )

        _, _, test_dataloader = get_real_data(
            np_test_data,
            batch_size,
            dataset_size,
            device,
            time_add_type=time_add_type,
            normalisation=normalisation,
            filter_by_time=filter_by_time,
            initial_point=False,
            scale=scale
        )

    elif data_type.lower() in ["gbm", "rbergomi"]:
        sdeint_kwargs = {
            "sde_method": gen_sde_method,
            "sde_dt_scale": gen_sde_dt_scale
        }

        if data_type.lower() == "gbm":
            sde_gen = sde.GeometricBrownianMotion(gen_sde_integration_type, "diagonal", *sde_parameters)
        elif data_type.lower() == "rbergomi":
            xi, eta, rho, H = sde_parameters
            sde_gen = rBergomi(n=int(path_length / end_time), N=dataset_size, T=end_time, a=H - 0.5, rho=rho, eta=eta, xi=xi)

        ts, data_size, train_dataloader = get_synthetic_data(
            sde_gen,
            batch_size,
            dataset_size,
            device,
            output_dim,
            path_length,
            normalisation=normalisation,
            scale=scale,
            sdeint_kwargs=sdeint_kwargs,
            end_time=end_time,
            time_add_type=time_add_type
        )

        _, _, test_dataloader = get_synthetic_data(
            sde_gen,
            batch_size,
            dataset_size,
            device,
            output_dim,
            path_length,
            normalisation=normalisation,
            scale=scale,
            sdeint_kwargs=sdeint_kwargs,
            end_time=end_time,
            time_add_type=time_add_type
        )

    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    transformer = Transformer(transformations, transformation_args, device).to(device) if do_transforms else lambda x: x

    return ts, data_size, train_dataloader, test_dataloader, infinite_train_dataloader, transformer

def train_model(config, device, generator, discriminator, generator_optimiser, discriminator_optimiser, 
                g_scheduler, d_scheduler, infinite_train_dataloader, transformer, ts,
                averaged_generator, averaged_discriminator, gen_fp, disc_fp):
    
    data_config = config["data_config"]
    optimizer_config = config["optimizer_config"]

    steps = optimizer_config["steps"]
    subtract_start = data_config["subtract_start"]
    early_stopping_type = optimizer_config["early_stopping_type"]
    crit_evals = optimizer_config["crit_evals"]
    crit_thresh = optimizer_config["crit_thresh"]
    cutoff = optimizer_config["cutoff"]
    ci = optimizer_config["mmd_ci"]
    mmd_periods = optimizer_config["mmd_periods"]
    swa_step_start = optimizer_config["swa_step_start"]
    adversarial = False # optimizer_config.get("adversarial", False)
    adapting_lr = optimizer_config["adapting_lr"]
    batch_size = next(infinite_train_dataloader)[0].shape[0] # expected size: (batch_size, seq_length, data_dim) 
    
    tr_loss = torch.zeros(steps, requires_grad=False).to(device)
    criterions = []
    burn_counter = 0

    trange = tqdm(range(steps), position=0, leave=True)

    for step in trange:
        ###############################################################################
        ## 1. Calculate loss
        ###############################################################################  
        
        real_samples, = next(infinite_train_dataloader)
        real_samples = transformer(real_samples).float()
        
        generated_samples = generator(ts, batch_size)
        generated_samples = transformer(generated_samples)

        if subtract_start:
            real_samples = subtract_initial_point(real_samples)
            generated_samples = subtract_initial_point(generated_samples)

        if isinstance(discriminator, CDEDiscriminator):
            gen_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = gen_score - real_score
        else:
            loss = discriminator(generated_samples, real_samples.detach())

        loss.backward()

        tr_loss[step] += loss.detach().clone()
        burn_counter += 1

        ###############################################################################
        ## 2. Step through optimisers and adapting LR schedulers, stochastic weights
        ###############################################################################
        if isinstance(discriminator, CDEDiscriminator):
            for param in discriminator.parameters():
                param.grad *= -1
            
            discriminator_optimiser.step()
            discriminator_optimiser.zero_grad()
            
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)
            
            if adapting_lr:
                d_scheduler.step()
            
        generator_optimiser.step()
        generator_optimiser.zero_grad()
        
        if adapting_lr:
            g_scheduler.step()

        if adversarial and step > swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)
            
        ###############################################################################
        ## 3. Early stopping criterions
        ###############################################################################
        if optimizer_config["early_stopping"] and ((step % optimizer_config["steps_per_print"]) == 0 or step == steps - 1):
            with torch.no_grad():
                if early_stopping_type == "marginals":
                    criterion = 0

                    for _ in range(crit_evals):
                        crit_generated_samples = generator(ts, batch_size)
                        criterion_samples, = next(infinite_train_dataloader)

                        if subtract_start:
                            crit_generated_samples = subtract_initial_point(crit_generated_samples)
                            criterion_samples = subtract_initial_point(criterion_samples)

                        criterion += stopping_criterion(criterion_samples, crit_generated_samples, cutoff=cutoff, print_results=False)
                    av_criterion = criterion / crit_evals
                    criterions.append(av_criterion)

                    if av_criterion > crit_thresh:
                        trange.write("Stopping criterion reached. Exiting training early.")
                        tr_loss[step:] = tr_loss[step]
                        break
                    crit_text = f"Criterion: {av_criterion:.4f} Target: {crit_thresh:.4f}"
                elif (early_stopping_type == "mmd") and step > mmd_periods:
                    averaged_mmd_score = torch.mean(tr_loss[step - mmd_periods:step])
                    if averaged_mmd_score <= ci:
                        trange.write("Stopping criterion reached. Exiting training early.")
                        tr_loss[step:] = tr_loss[step]
                        break
                    crit_text = f"Criterion: {averaged_mmd_score:.5e} Target {ci:.5e}"
                else:
                    crit_text = ""

                total_unaveraged_loss = loss.item()

                if step > swa_step_start:
                    total_averaged_loss = total_unaveraged_loss
                    trange.write(f"Step: {step:3} Total loss (unaveraged): {total_unaveraged_loss:.5e} "
                                 f"Loss (averaged): {total_averaged_loss:.5e} " + crit_text)
                else:
                    trange.write(f"Step: {step:3} Total loss (unaveraged): {total_unaveraged_loss:.5e} " + crit_text)
                
    ###############################################################################
    ## 4. Training complete
    ###############################################################################
        if ((step % optimizer_config["steps_per_print"]) == 0 or step == steps - 1):
            torch.save(generator.state_dict(), gen_fp + f"_step{step}.pkl")
            torch.save(discriminator.state_dict(), disc_fp + f"_step{step}.pkl")
            torch.save(averaged_generator.state_dict(), gen_fp + f"_step{step}_averaged.pkl")
            torch.save(averaged_discriminator.state_dict(), disc_fp + f"_step{step}_averaged.pkl")
    torch.save(generator.state_dict(), gen_fp + ".pkl")
    torch.save(discriminator.state_dict(), disc_fp + ".pkl")
    torch.save(config["generator_config"], gen_fp + "_config.pkl")
    torch.save(config[discriminator.__class__.__name__ + "_config"], disc_fp + "_config.pkl")
    torch.save(averaged_generator.state_dict(), gen_fp + "_averaged.pkl")
    torch.save(averaged_discriminator.state_dict(), disc_fp + "_averaged.pkl")
    return tr_loss
